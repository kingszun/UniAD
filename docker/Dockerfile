FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# =============================================================================
# UniAD v2.0 stack (KAK-22)
# - base: nvidia/cuda 11.8 cudnn8 runtime ubuntu22.04 (matches gameformer fork)
# - python 3.9 via uv-managed CPython at /opt/uv-python (INSTALL.md spec)
# - venv: /opt/venv/uniad
# - PATH: venv/bin first → python/pip auto-resolve to venv binary
#
# Dual-mode (gameformer 패턴):
# - USER root by default. compose 가 user 1000 으로 override 시 host 파일 권한 일치.
# - RunPod pod 에서는 root 로 진입 → sshd, /root/.ssh/authorized_keys, /workspace mount.
# - ENTRYPOINT root branch 가 PUBLIC_KEY → authorized_keys + sshd 기동. uid 1000 path 는 skip.
# =============================================================================
USER root

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX" \
    CMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
    UV_PROJECT_ENVIRONMENT=/opt/venv/uniad \
    UV_PYTHON=3.9 \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/uniad/bin:/usr/local/bin:/usr/bin:/bin

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        gnupg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg \
        openssh-server \
        iproute2 \
    && mkdir -p /var/run/sshd /root/.ssh \
    && chmod 700 /root/.ssh \
    && sed -i 's/^#*PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv

WORKDIR /workspace/UniAD

# venv with python 3.9 (INSTALL.md spec)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv ${UV_PROJECT_ENVIRONMENT} --python 3.9

# torch 2.0.1 + cu118 (INSTALL.md)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${UV_PROJECT_ENVIRONMENT}/bin/python \
        torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
        --index-url https://download.pytorch.org/whl/cu118

# mmcv-full 1.6.1 (prebuilt cu118 + torch 2.0 wheel, no source build)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${UV_PROJECT_ENVIRONMENT}/bin/python \
        mmcv-full==1.6.1 \
        -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

# mm-series
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${UV_PROJECT_ENVIRONMENT}/bin/python \
        mmdet==2.26.0 \
        mmsegmentation==0.29.1 \
        mmdet3d==1.0.0rc6

# UniAD project requirements
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${UV_PROJECT_ENVIRONMENT}/bin/python \
        -r /tmp/requirements.txt

# ssh session 도 venv PATH 받도록 (gameformer KAK-17 패턴):
# Docker ENV 는 PID 1 만 적용 — sshd 가 fork 한 새 session 은 안 받음.
# - /etc/environment: PAM 이 모든 session 시작 시 load (login + ssh non-login non-interactive 모두)
# - /etc/bash.bashrc: interactive bash 추가 보장
RUN printf 'PATH="/opt/venv/uniad/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"\n' > /etc/environment \
    && printf '\nexport PATH=/opt/venv/uniad/bin:$PATH\n' >> /etc/bash.bashrc

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["sleep", "infinity"]
