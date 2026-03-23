#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG=${IMAGE_TAG:-flashattn-cu128:py310}
CUDA_VERSION=${CUDA_VERSION:-12.8.0}
UBUNTU_VERSION=${UBUNTU_VERSION:-22.04}
TORCH_VERSION=${TORCH_VERSION:-2.10.0}
TORCH_CUDA=${TORCH_CUDA:-cu128}
FLASH_ATTN_VERSION=${FLASH_ATTN_VERSION:-2.8.3}
TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
DEFAULT_JOBS=8
if command -v nproc >/dev/null 2>&1; then
  NPROC=$(nproc)
  if [ "$NPROC" -lt "$DEFAULT_JOBS" ]; then
    DEFAULT_JOBS="$NPROC"
  fi
fi
MAX_JOBS=${MAX_JOBS:-$DEFAULT_JOBS}

PLATFORM_ARGS=()
if [ -n "${DOCKER_PLATFORM:-}" ]; then
  PLATFORM_ARGS+=(--platform "${DOCKER_PLATFORM}")
fi

docker build "${PLATFORM_ARGS[@]}" -f docker/Dockerfile.flashattn \
  --build-arg CUDA_VERSION="${CUDA_VERSION}" \
  --build-arg UBUNTU_VERSION="${UBUNTU_VERSION}" \
  --build-arg TORCH_VERSION="${TORCH_VERSION}" \
  --build-arg TORCH_CUDA="${TORCH_CUDA}" \
  --build-arg FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION}" \
  --build-arg TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
  --build-arg MAX_JOBS="${MAX_JOBS}" \
  -t "${IMAGE_TAG}" .
