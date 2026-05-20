# Build stage: CUDA devel image needed to compile detectron2's CUDA extensions
# torch 2.11 targets CUDA 13.0 (nvidia-cuda-* pip packages are 13.0.x); 13.1.2 is backward compatible
FROM nvidia/cuda:13.1.2-cudnn-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    ninja-build gcc g++ git && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build
COPY pyproject.toml uv.lock ./

# FORCE_CUDA=1 pre-compiles detectron2 CUDA ops at install time instead of lazily at first import.
# TORCH_CUDA_ARCH_LIST must be set explicitly — without a GPU in the build environment,
# torch's auto-detection returns an empty list and crashes on arch_list[-1].
# Adjust to match your actual target hardware to keep the binary smaller.
ENV UV_PROJECT_ENVIRONMENT=/env \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0"
RUN uv sync --frozen --no-install-project && \
    # Strip debug symbols from compiled extensions (~500MB–2GB savings)
    find /env -name "*.so" -exec strip --strip-debug {} \; 2>/dev/null || true && \
    # Remove test suites bundled inside packages
    find /env -type d \( -name "test" -o -name "tests" \) -exec rm -rf {} + 2>/dev/null || true


# Runtime stage: torch pip wheels bundle CUDA runtime via nvidia-* packages — no CUDA base needed
FROM python:3.12-slim-bookworm AS runner

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 libegl1 libopengl0 && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /env /env

# Builder (ubuntu24.04) has python at /usr/bin/python3.12;
# runtime (bookworm) has it at /usr/local/bin/python3.12 — fix the venv symlinks.
RUN ln -sf /usr/local/bin/python3.12 /env/bin/python && \
    ln -sf /usr/local/bin/python3.12 /env/bin/python3 && \
    ln -sf /usr/local/bin/python3.12 /env/bin/python3.12

ENV PATH=/env/bin:$PATH \
    VIRTUAL_ENV=/env \
    PYTHONPATH=/src/laypa

RUN useradd --uid 1000 --no-create-home --shell /bin/bash laypa && \
    mkdir -p /.gunicorn && chown 1000:1000 /.gunicorn

WORKDIR /src/laypa
COPY . .

ARG GIT_HASH=unknown
RUN echo "${GIT_HASH}" > version_info

COPY docker/_entrypoint_uv.sh /_entrypoint.sh
ENTRYPOINT ["/_entrypoint.sh"]
CMD ["/bin/bash"]

USER 1000
