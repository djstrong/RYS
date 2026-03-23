#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG=${IMAGE_TAG:-rys-exllama-precompiled:cu128}
FLASH_ATTN_IMAGE=${FLASH_ATTN_IMAGE:-flashattn-cu128:py310}
EXLLAMAV3_PATH=${EXLLAMAV3_PATH:-/home/grace/exllamav3}
DOCKER_PLATFORM=${DOCKER_PLATFORM:-}
TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
MAX_JOBS=${MAX_JOBS:-4}

DOCKER_BUILDKIT=1 docker build \
  -f docker/Dockerfile.exllama_precompiled \
  --build-arg BASE_IMAGE="$FLASH_ATTN_IMAGE" \
  --build-arg TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  --build-arg MAX_JOBS="$MAX_JOBS" \
  --build-context exllamav3="$EXLLAMAV3_PATH" \
  ${DOCKER_PLATFORM:+--platform "$DOCKER_PLATFORM"} \
  -t "$IMAGE_TAG" .
