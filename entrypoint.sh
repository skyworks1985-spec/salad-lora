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
#   4. uploads remaining outputs and the training log at the end.
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

NETWORK_DIM="${NETWORK_DIM:-32}"
NETWORK_ALPHA="${NETWORK_ALPHA:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-100}"
RESOLUTION="${RESOLUTION:-1024}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
SEED="${SEED:-42}"
CACHE_LATENTS_TO_DISK="${CACHE_LATENTS_TO_DISK:-0}"  # 1 = also write latent cache to disk

# Upload target configuration. UPLOAD_MODE: put | rclone | none
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
# TODO(storage): the file host for model/dataset/results is not decided yet.
#   - "put"    expects UPLOAD_PUT_BASE_URL to accept HTTP PUT of a raw file
#              (presigned URL base, or a token-protected endpoint via
#              UPLOAD_AUTH_HEADER). The file name is appended to the base.
#   - "rclone" expects RCLONE_REMOTE like "remote:bucket/prefix" and an image
#              that ships rclone plus its config (not in the current image).
#   Decide the host, then verify this function end to end in the smoke test.
upload_file() {
  local file="$1"
  local name
  name="$(basename "$file")"
  case "$UPLOAD_MODE" in
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
# -C - resumes a partial download after a node interruption restarts the job.
curl -fL --retry 5 --retry-all-errors -C - -o "$MODEL_FILE" "$MODEL_URL" || {
  log "ERROR: model download failed"; exit 2; }
ls -lh "$MODEL_FILE"

marker "DOWNLOAD dataset"
curl -fL --retry 5 --retry-all-errors -o "$DATASET_ZIP" "$DATASET_ZIP_URL" || {
  log "ERROR: dataset download failed"; exit 2; }
unzip -q -o "$DATASET_ZIP" -d "$EXTRACT_DIR" || {
  log "ERROR: dataset unzip failed"; exit 2; }
TRAIN_DATA_DIR="$(detect_train_dir "$EXTRACT_DIR")"
IMG_COUNT="$(find "$TRAIN_DATA_DIR" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \) | wc -l)"
log "train_data_dir=$TRAIN_DATA_DIR images=$IMG_COUNT"

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
