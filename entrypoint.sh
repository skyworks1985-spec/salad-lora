#!/usr/bin/env bash
# SaladCloud LoRA training entrypoint (Kohya sd-scripts, SDXL).
#
# SaladCloud containers have no SSH and no persistent storage, and nodes are
# interruptible home PCs. This script therefore:
#   1. downloads the base model and dataset zip at start (env: MODEL_URL,
#      DATASET_ZIP_URL),
#   2. runs training, streaming logs to stdout (visible via the Salad logs API),
#   3. runs a background watcher that uploads every epoch checkpoint the
#      moment it appears, so an interruption loses at most the current epoch,
#   4. uploads remaining outputs and the training log at the end,
#   5. optionally ships the console log (run.log) for outside visibility:
#      to ntfy.sh when LOG_TOPIC is set (0.3.0), and as a live snapshot to
#      outputs/<OUTPUT_NAME>/run.log on HF when UPLOAD_MODE=hf (0.4.0),
#      because the Salad logs API returns 404 for this project.
#
# Training parameters mirror the known-good local recipe (see README.md).
# Exit code: 0 on training success, 2 on configuration error, otherwise the
# training process exit code.

set -uo pipefail

# ---------------------------------------------------------------------------
# Configuration (environment variables, with recipe defaults)
# ---------------------------------------------------------------------------
MODEL_URL="${MODEL_URL:-}"                 # required: base model .safetensors
DATASET_ZIP_URL="${DATASET_ZIP_URL:-}"     # required: dataset .zip (kohya folder layout)
OUTPUT_NAME="${OUTPUT_NAME:-lora}"         # output file base name / trigger folder set name
EPOCHS="${EPOCHS:-5}"                      # epochs (ignored when MAX_TRAIN_STEPS is set)
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"     # optional step cap, e.g. 100 for the smoke test
DIAG_MODE="${DIAG_MODE:-}"                   # "survival" = downloads only, heartbeat, no training
DIAG_MINUTES="${DIAG_MINUTES:-30}"           # survival-mode heartbeat duration, in minutes
LOG_TOPIC="${LOG_TOPIC:-}"                   # if set, ship run.log to ntfy.sh/$LOG_TOPIC (log visibility)
HF_SHIP_INTERVAL="${HF_SHIP_INTERVAL:-30}"   # seconds between HF run.log snapshot uploads (UPLOAD_MODE=hf)

NETWORK_DIM="${NETWORK_DIM:-32}"
NETWORK_ALPHA="${NETWORK_ALPHA:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-100}"
RESOLUTION="${RESOLUTION:-1024}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
SEED="${SEED:-42}"
CACHE_LATENTS_TO_DISK="${CACHE_LATENTS_TO_DISK:-0}"  # 1 = also write latent cache to disk

# Hugging Face storage (decided 2026-11-11): private repo Works403/lora-storage
# holds models/, datasets/ and outputs/. Downloads of the private repo need
# HF_TOKEN; UPLOAD_MODE=hf uploads results back into outputs/<OUTPUT_NAME>/.
HF_TOKEN="${HF_TOKEN:-}"                        # HF access token (write)
HF_REPO="${HF_REPO:-Works403/lora-storage}"     # storage repo id

# Upload target configuration. UPLOAD_MODE: hf | put | rclone | none
UPLOAD_MODE="${UPLOAD_MODE:-none}"
UPLOAD_PUT_BASE_URL="${UPLOAD_PUT_BASE_URL:-}"  # PUT base URL, file name is appended
UPLOAD_AUTH_HEADER="${UPLOAD_AUTH_HEADER:-}"    # optional, e.g. "Authorization: Bearer xxx"
RCLONE_REMOTE="${RCLONE_REMOTE:-}"              # e.g. "remote:bucket/prefix" (needs rclone in image)

WATCH_INTERVAL="${WATCH_INTERVAL:-15}"          # seconds between checkpoint scans

WORK_ROOT="${WORK_ROOT:-/workspace}"
SD_SCRIPTS_DIR="${SD_SCRIPTS_DIR:-/opt/sd-scripts}"
MODEL_FILE="$WORK_ROOT/model/base_model.safetensors"
DATASET_ZIP="$WORK_ROOT/dataset.zip"
EXTRACT_DIR="$WORK_ROOT/extract"
OUTPUT_DIR="$WORK_ROOT/output"
LOG_FILE="$WORK_ROOT/train.log"
RUN_LOG="$WORK_ROOT/run.log"
SHIP_OFFSET_FILE="$WORK_ROOT/.ship_offset"
SHIP_STOP_FILE="$WORK_ROOT/.ship_stop"
SHIP_CHUNK_MAX=3500
SHIPPER_PID=""
HF_SHIP_MARK_FILE="$WORK_ROOT/.hf_ship_mark"
HF_SHIP_STOP_FILE="$WORK_ROOT/.hf_ship_stop"
HF_SHIP_PID=""
UPLOAD_MANIFEST="$WORK_ROOT/.uploaded"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()    { echo "[salad-lora $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
marker() { echo "===== [salad-lora] $* ====="; }

# upload_file <path>
# Pluggable single-file upload. Returns 0 when the file is safely stored
# (or when UPLOAD_MODE=none, in which case it only logs the intent).
#
# Storage is Hugging Face (decided 2026-11-11): UPLOAD_MODE=hf sends the file
# to outputs/<OUTPUT_NAME>/ in HF_REPO via /opt/hf_put.py (Git LFS flow).
# "put" and "rclone" remain as fallbacks for other hosts.
upload_file() {
  local file="$1"
  local name
  name="$(basename "$file")"
  case "$UPLOAD_MODE" in
    hf)
      if [ -z "$HF_TOKEN" ]; then
        log "ERROR: UPLOAD_MODE=hf but HF_TOKEN is empty; cannot upload $name"
        return 1
      fi
      python3 /opt/hf_put.py "$file" "outputs/$OUTPUT_NAME/$name" \
        "Upload $name ($OUTPUT_NAME)"
      ;;
    put)
      if [ -z "$UPLOAD_PUT_BASE_URL" ]; then
        log "ERROR: UPLOAD_MODE=put but UPLOAD_PUT_BASE_URL is empty; cannot upload $name"
        return 1
      fi
      local -a auth=()
      if [ -n "$UPLOAD_AUTH_HEADER" ]; then auth=(-H "$UPLOAD_AUTH_HEADER"); fi
      curl -fsS --retry 3 --retry-delay 5 -T "$file" "${auth[@]}" \
        "${UPLOAD_PUT_BASE_URL%/}/$name"
      ;;
    rclone)
      if [ -z "$RCLONE_REMOTE" ] || ! command -v rclone >/dev/null 2>&1; then
        log "ERROR: UPLOAD_MODE=rclone but RCLONE_REMOTE is empty or rclone is missing; cannot upload $name"
        return 1
      fi
      rclone copyto "$file" "${RCLONE_REMOTE%/}/$name"
      ;;
    none|*)
      log "upload skipped (UPLOAD_MODE=$UPLOAD_MODE): $name"
      ;;
  esac
}

already_uploaded() {
  [ -f "$UPLOAD_MANIFEST" ] && grep -qxF "$1" "$UPLOAD_MANIFEST"
}

# Upload every finished *.safetensors in OUTPUT_DIR that is not yet uploaded.
# A file counts as finished when its size is stable across a 5 second window.
upload_pending_checkpoints() {
  local f name s1 s2
  for f in "$OUTPUT_DIR"/*.safetensors; do
    [ -e "$f" ] || continue
    name="$(basename "$f")"
    already_uploaded "$name" && continue
    s1="$(stat -c%s "$f")"
    sleep 5
    s2="$(stat -c%s "$f")"
    if [ "$s1" != "$s2" ]; then
      log "checkpoint still being written, will retry: $name"
      continue
    fi
    log "uploading checkpoint: $name ($s2 bytes)"
    if upload_file "$f"; then
      echo "$name" >> "$UPLOAD_MANIFEST"
      log "checkpoint uploaded: $name"
    else
      log "ERROR: upload failed, will retry: $name"
    fi
  done
}

watcher_loop() {
  while true; do
    upload_pending_checkpoints
    sleep "$WATCH_INTERVAL"
  done
}

# Find the directory that directly contains kohya repeat folders (e.g.
# "10_yoshino_char"). Falls back to the extract root.
detect_train_dir() {
  local root="$1" d
  if find "$root" -maxdepth 1 -mindepth 1 -type d -regextype posix-extended \
       -regex '.*/[0-9]+_[^/]+' | grep -q .; then
    echo "$root"; return
  fi
  for d in "$root"/*/; do
    [ -d "$d" ] || continue
    if find "$d" -maxdepth 1 -mindepth 1 -type d -regextype posix-extended \
         -regex '.*/[0-9]+_[^/]+' | grep -q .; then
      echo "${d%/}"; return
    fi
  done
  echo "$root"
}

# ---------------------------------------------------------------------------
# Log shipping (ntfy.sh) — diagnostic log visibility (added in 0.3.0)
#
# Salad's logs API returns 404 for this project, so when LOG_TOPIC is set we
# mirror the console log to a public ntfy.sh topic (readable only by someone
# who knows the unguessable topic name). All script output is tee'd into
# $RUN_LOG; a background shipper posts the not-yet-sent tail every 30s, and
# immediately when a new phase marker appears. Everything posted is scrubbed
# of URL query strings so signed URLs (which carry credentials) never leak;
# no secret values are printed to the log in the first place. On exit, a
# trap ships the final tail plus "FINAL rc=<code>". Every shipper step is
# best-effort (|| true): a shipper failure must never affect the main flow.
# With LOG_TOPIC unset, behavior is identical to 0.2.0.
# ---------------------------------------------------------------------------
scrub_log_text() {
  # Strip the query string ("?" and everything after it) from URLs.
  sed -E 's#(https?://[^?[:space:]]+)\?[^[:space:]]*#\1#g'
}

ntfy_post() {
  [ -n "$LOG_TOPIC" ] || return 0
  curl -sS -m 20 -X POST --data-binary "$1" "https://ntfy.sh/$LOG_TOPIC" \
    >/dev/null 2>&1 || true
}

ship_new_log() {
  # Post the not-yet-sent tail of $RUN_LOG, scrubbed, in <=3.5KB chunks.
  # Offset bookkeeping uses raw $RUN_LOG bytes; scrubbing only alters content.
  [ -n "$LOG_TOPIC" ] || return 0
  [ -f "$RUN_LOG" ] || return 0
  local size=0 sent=0
  size="$(stat -c%s "$RUN_LOG" 2>/dev/null || echo 0)"
  [ -f "$SHIP_OFFSET_FILE" ] && sent="$(cat "$SHIP_OFFSET_FILE" 2>/dev/null || echo 0)"
  case "$sent" in ''|*[!0-9]*) sent=0 ;; esac
  [ "$size" -gt "$sent" ] || return 0
  local tmp="$WORK_ROOT/.ship_new"
  tail -c +$(( sent + 1 )) "$RUN_LOG" 2>/dev/null | scrub_log_text > "$tmp" || true
  echo "$size" > "$SHIP_OFFSET_FILE" || true
  local total=0 chunks=0 i=0 part=""
  total="$(stat -c%s "$tmp" 2>/dev/null || echo 0)"
  if [ "$total" -le 0 ]; then rm -f "$tmp"; return 0; fi
  chunks=$(( (total + SHIP_CHUNK_MAX - 1) / SHIP_CHUNK_MAX ))
  for i in $(seq 1 "$chunks"); do
    part="$(tail -c +$(( (i - 1) * SHIP_CHUNK_MAX + 1 )) "$tmp" 2>/dev/null | head -c "$SHIP_CHUNK_MAX")"
    [ -n "$part" ] && ntfy_post "$part"
  done
  rm -f "$tmp"
  return 0
}

unsent_has_marker() {
  # True when the unsent tail of $RUN_LOG contains a phase marker line.
  [ -f "$RUN_LOG" ] || return 1
  local size=0 sent=0
  size="$(stat -c%s "$RUN_LOG" 2>/dev/null || echo 0)"
  [ -f "$SHIP_OFFSET_FILE" ] && sent="$(cat "$SHIP_OFFSET_FILE" 2>/dev/null || echo 0)"
  case "$sent" in ''|*[!0-9]*) sent=0 ;; esac
  [ "$size" -gt "$sent" ] || return 1
  tail -c +$(( sent + 1 )) "$RUN_LOG" 2>/dev/null | grep -q '===== \[salad-lora\]'
}

shipper_loop() {
  local last=0 now=0
  last="$(date +%s)"
  while [ ! -f "$SHIP_STOP_FILE" ]; do
    sleep 5
    now="$(date +%s)"
    if unsent_has_marker; then
      ship_new_log || true
      last="$(date +%s)"
    elif [ $(( now - last )) -ge 30 ]; then
      ship_new_log || true
      last="$now"
    fi
  done
}

# ---------------------------------------------------------------------------
# HF log shipping — diagnostic log visibility (added in 0.4.0)
#
# When UPLOAD_MODE=hf, a background shipper snapshots $RUN_LOG to
# outputs/<OUTPUT_NAME>/run.log in HF_REPO every HF_SHIP_INTERVAL seconds
# (and immediately when a new phase marker appears), overwriting the same
# path via /opt/hf_put.py. An outside reader can therefore follow the run
# live through the HF connector, and the last snapshot survives a sudden
# node loss. While HF_TOKEN is empty the shipper waits quietly (one note,
# no repeated upload attempts): the user pastes the token into the group's
# env in the Salad portal, Salad redeploys the container, and the new
# process starts shipping immediately. On exit, a trap uploads the final
# snapshot, and the final sweep in Main uploads run.log once more. Every
# step is best-effort (|| true): a shipper failure must never affect the
# main flow. With UPLOAD_MODE != hf and LOG_TOPIC unset, behavior is
# identical to 0.2.0.
# ---------------------------------------------------------------------------
hf_ship_snapshot() {
  # Upload one snapshot of $RUN_LOG to outputs/$OUTPUT_NAME/run.log.
  # Records the snapshot size in HF_SHIP_MARK_FILE on success.
  [ "$UPLOAD_MODE" = "hf" ] || return 0
  [ -n "$HF_TOKEN" ] || return 0
  [ -f "$RUN_LOG" ] || return 0
  local snap="$WORK_ROOT/.runlog_snapshot" sz=0
  cp "$RUN_LOG" "$snap" 2>/dev/null || return 0
  sz="$(stat -c%s "$snap" 2>/dev/null || echo 0)"
  if python3 /opt/hf_put.py "$snap" "outputs/$OUTPUT_NAME/run.log" \
       "Update run.log ($OUTPUT_NAME)" >/dev/null 2>&1; then
    echo "$sz" > "$HF_SHIP_MARK_FILE" || true
    return 0
  fi
  return 1
}

hf_unsent_has_marker() {
  # True when the part of $RUN_LOG beyond the last shipped snapshot
  # contains a phase marker line.
  [ -f "$RUN_LOG" ] || return 1
  local size=0 sent=0
  size="$(stat -c%s "$RUN_LOG" 2>/dev/null || echo 0)"
  [ -f "$HF_SHIP_MARK_FILE" ] && sent="$(cat "$HF_SHIP_MARK_FILE" 2>/dev/null || echo 0)"
  case "$sent" in ''|*[!0-9]*) sent=0 ;; esac
  [ "$size" -gt "$sent" ] || return 1
  tail -c +$(( sent + 1 )) "$RUN_LOG" 2>/dev/null | grep -q '===== \[salad-lora\]'
}

hf_shipper_loop() {
  local last=0 now=0 size=0 sent=0
  last="$(date +%s)"
  if [ -z "$HF_TOKEN" ]; then
    # The token is pasted into the group env in the Salad portal and only
    # reaches the container on the next (re)deploy, so just wait quietly:
    # note it once and keep retrying without attempting uploads.
    log "HF log shipper: HF_TOKEN not set; waiting quietly (add it via the Salad portal env and the container will ship run.log after redeploy)"
  fi
  while [ ! -f "$HF_SHIP_STOP_FILE" ]; do
    sleep 5
    [ -n "$HF_TOKEN" ] || continue
    now="$(date +%s)"
    size="$(stat -c%s "$RUN_LOG" 2>/dev/null || echo 0)"
    sent=0
    [ -f "$HF_SHIP_MARK_FILE" ] && sent="$(cat "$HF_SHIP_MARK_FILE" 2>/dev/null || echo 0)"
    case "$sent" in ''|*[!0-9]*) sent=0 ;; esac
    [ "$size" -gt "$sent" ] || continue
    if hf_unsent_has_marker || [ $(( now - last )) -ge "$HF_SHIP_INTERVAL" ]; then
      hf_ship_snapshot || true
      last="$(date +%s)"
    fi
  done
}

on_exit_ship() {
  local rc=$?
  if [ -n "$LOG_TOPIC" ] || [ "$UPLOAD_MODE" = "hf" ]; then
    sleep 2 || true   # give tee a moment to flush the last lines into run.log
  fi
  if [ "$UPLOAD_MODE" = "hf" ]; then
    # Stop the shipper loop first so it cannot race the final snapshot.
    touch "$HF_SHIP_STOP_FILE" || true
    [ -n "$HF_SHIP_PID" ] && kill "$HF_SHIP_PID" >/dev/null 2>&1 || true
    hf_ship_snapshot || true
  fi
  if [ -n "$LOG_TOPIC" ]; then
    ship_new_log || true
    ntfy_post "FINAL rc=$rc" || true
    [ -n "$SHIPPER_PID" ] && kill "$SHIPPER_PID" >/dev/null 2>&1 || true
  fi
  return 0
}

# Log collection (tee into $RUN_LOG) and the shippers run when either
# channel is enabled: ntfy (LOG_TOPIC) or HF (UPLOAD_MODE=hf).
LOG_COLLECT=0
[ -n "$LOG_TOPIC" ] && LOG_COLLECT=1
[ "$UPLOAD_MODE" = "hf" ] && LOG_COLLECT=1
if [ "$LOG_COLLECT" = "1" ]; then
  mkdir -p "$WORK_ROOT" || true
  rm -f "$SHIP_STOP_FILE" "$HF_SHIP_STOP_FILE" || true
  echo 0 > "$SHIP_OFFSET_FILE" || true
  echo 0 > "$HF_SHIP_MARK_FILE" || true
  exec > >(tee -a "$RUN_LOG") 2>&1
  trap on_exit_ship EXIT
  if [ -n "$LOG_TOPIC" ]; then
    shipper_loop &
    SHIPPER_PID=$!
  fi
  if [ "$UPLOAD_MODE" = "hf" ]; then
    hf_shipper_loop &
    HF_SHIP_PID=$!
  fi
fi

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
marker "START output_name=$OUTPUT_NAME epochs=$EPOCHS max_steps=${MAX_TRAIN_STEPS:-none} upload_mode=$UPLOAD_MODE"
nvidia-smi || log "WARNING: nvidia-smi not available"

if [ -z "$MODEL_URL" ] || [ -z "$DATASET_ZIP_URL" ]; then
  log "ERROR: MODEL_URL and DATASET_ZIP_URL must both be set"
  exit 2
fi

mkdir -p "$(dirname "$MODEL_FILE")" "$EXTRACT_DIR" "$OUTPUT_DIR"
touch "$UPLOAD_MANIFEST"

marker "DOWNLOAD base model"
# Private HF repo downloads need the token; public URLs work without it.
declare -a DL_AUTH=()
if [ -n "$HF_TOKEN" ]; then DL_AUTH=(-H "Authorization: Bearer $HF_TOKEN"); fi
# -C - resumes a partial download after a node interruption restarts the job.
curl -sS -fL --retry 5 --retry-all-errors -C - -o "$MODEL_FILE" "${DL_AUTH[@]}" "$MODEL_URL" || {
  log "ERROR: model download failed"; exit 2; }
ls -lh "$MODEL_FILE"

marker "DOWNLOAD dataset"
curl -sS -fL --retry 5 --retry-all-errors -o "$DATASET_ZIP" "${DL_AUTH[@]}" "$DATASET_ZIP_URL" || {
  log "ERROR: dataset download failed"; exit 2; }
unzip -q -o "$DATASET_ZIP" -d "$EXTRACT_DIR" || {
  log "ERROR: dataset unzip failed"; exit 2; }
TRAIN_DATA_DIR="$(detect_train_dir "$EXTRACT_DIR")"
IMG_COUNT="$(find "$TRAIN_DATA_DIR" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \) | wc -l)"
log "train_data_dir=$TRAIN_DATA_DIR images=$IMG_COUNT"

# ---------------------------------------------------------------------------
# Diagnostic survival mode (DIAG_MODE=survival)
#
# Downloads finish exactly as in a normal run, but instead of training the
# container heartbeats for DIAG_MINUTES and then exits 0 on its own. This
# separates Salad node instability from training-side crashes: if an instance
# cannot survive even this, the node side is at fault; if it survives the
# full window, the training path is the suspect.
# ---------------------------------------------------------------------------
if [ "$DIAG_MODE" = "survival" ]; then
  marker "DIAG MODE survival: downloads done, training skipped (heartbeat ${DIAG_MINUTES} min)"
  DIAG_START="$(date +%s)"
  DIAG_END=$(( DIAG_START + DIAG_MINUTES * 60 ))
  while [ "$(date +%s)" -lt "$DIAG_END" ]; do
    log "DIAG heartbeat: alive, elapsed $(( $(date +%s) - DIAG_START ))s of $(( DIAG_MINUTES * 60 ))s"
    sleep 60
  done
  marker "DIAG SURVIVED the full ${DIAG_MINUTES} minutes; exiting 0"
  exit 0
fi

marker "START checkpoint watcher (interval ${WATCH_INTERVAL}s)"
watcher_loop &
WATCH_PID=$!

marker "TRAIN"
TRAIN_ARGS=(
  --pretrained_model_name_or_path="$MODEL_FILE"
  --train_data_dir="$TRAIN_DATA_DIR"
  --output_dir="$OUTPUT_DIR"
  --output_name="$OUTPUT_NAME"
  --resolution="${RESOLUTION},${RESOLUTION}"
  --enable_bucket
  --caption_extension=".txt"
  --network_module="networks.lora"
  --network_dim="$NETWORK_DIM"
  --network_alpha="$NETWORK_ALPHA"
  --network_train_unet_only
  --learning_rate="$LEARNING_RATE"
  --lr_scheduler="$LR_SCHEDULER"
  --lr_warmup_steps="$LR_WARMUP_STEPS"
  --optimizer_type="adamw8bit"
  --mixed_precision="bf16"
  --save_precision="fp16"
  --sdpa
  --gradient_checkpointing
  --cache_latents
  --train_batch_size="$TRAIN_BATCH_SIZE"
  --save_every_n_epochs=1
  --seed="$SEED"
  --max_data_loader_n_workers=2
)
if [ "$CACHE_LATENTS_TO_DISK" = "1" ]; then
  TRAIN_ARGS+=(--cache_latents_to_disk)
fi
if [ -n "$MAX_TRAIN_STEPS" ]; then
  TRAIN_ARGS+=(--max_train_steps="$MAX_TRAIN_STEPS")
else
  TRAIN_ARGS+=(--max_train_epochs="$EPOCHS")
fi
# NOTE: --cache_text_encoder_outputs is intentionally absent. It conflicts
# with text-encoder training state expectations; the local recipe relies on
# --network_train_unet_only instead (see README.md).

cd "$SD_SCRIPTS_DIR"
python3 sdxl_train_network.py "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
TRAIN_RC="${PIPESTATUS[0]}"
cd "$WORK_ROOT"

marker "TRAIN exited with code $TRAIN_RC; final upload sweep"
upload_pending_checkpoints
kill "$WATCH_PID" >/dev/null 2>&1 || true
upload_file "$LOG_FILE" || log "WARNING: log upload failed"
if [ -f "$RUN_LOG" ]; then
  upload_file "$RUN_LOG" || log "WARNING: run.log upload failed"
fi
for f in "$OUTPUT_DIR"/*.safetensors; do
  [ -e "$f" ] || continue
  if ! already_uploaded "$(basename "$f")"; then
    log "final upload attempt: $(basename "$f")"
    upload_file "$f" || log "WARNING: could not upload $(basename "$f")"
  fi
done

if [ "$TRAIN_RC" -eq 0 ]; then
  marker "DONE success"
else
  marker "DONE training failed (code $TRAIN_RC)"
fi
exit "$TRAIN_RC"
