# SaladCloud LoRA training image
# Kohya sd-scripts (SDXL LoRA) on CUDA 12.4 / PyTorch 2.6.0, aimed at RTX 4090 (Ada, sm_89).
#
# Build:  docker build -t <registry>/salad-lora-train:0.1.0 .
# Run:    driven entirely by environment variables, see entrypoint.sh / README.md.

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# sd-scripts release pin. Bump deliberately and rebuild; do not float on main.
# v0.12.3 is the release current as of 2026-11-11. Its README prescribes
# torch 2.6.0 / torchvision 0.21.0 (cu124), which the base image provides.
ARG SD_SCRIPTS_VERSION=v0.12.3

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WORK_ROOT=/workspace

# git: clone sd-scripts at build time. curl/unzip: entrypoint downloads the
# base model and the dataset zip at container start.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git curl unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${SD_SCRIPTS_VERSION}" \
        https://github.com/kohya-ss/sd-scripts.git /opt/sd-scripts

WORKDIR /opt/sd-scripts

# sd-scripts requirements.txt pins the training stack (accelerate, diffusers,
# transformers, ...). bitsandbytes provides the adamw8bit optimizer; it is
# normally pulled in by requirements.txt, the explicit install is a belt-and-
# suspenders no-op when it is already present.
RUN pip install -r requirements.txt \
    && pip install bitsandbytes \
    && python -c "import torch, bitsandbytes; print('torch', torch.__version__, '| cuda', torch.version.cuda)"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
