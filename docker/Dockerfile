FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# =============================================================================
# UniAD v1 stack (KAK-22, KAK-35).
# - INSTALL.md: python 3.8 / cuda 11.1 / torch 1.9.1+cu111 / mmcv-full 1.4.0
#   / mmdet 2.14.0 / mmsegmentation 0.14.1 / mmdet3d v0.17.1 (source build)
# - Devel base — mmdet3d v0.17.1 source build 시 nvcc 필요.
# - gameformer 패턴: uv venv, sshd, dual-mode entrypoint.
#
# Dual-mode:
# - USER root by default. compose 가 user 1000 으로 override 시 host 파일 권한 일치.
# - RunPod pod 에서는 root → sshd, /root/.ssh/authorized_keys, /workspace mount.
# - ENTRYPOINT root branch 만 PUBLIC_KEY → authorized_keys + sshd 기동.
# =============================================================================
USER root

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    TZ=Asia/Seoul \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,display \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6+PTX" \
    UV_PROJECT_ENVIRONMENT=/opt/venv/uniad \
    UV_PYTHON=3.8 \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/uniad/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin

# old nvidia repo gpg keys (cuda 11.1 image — apt-get update gpg error 회피)
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        curl \
        ca-certificates \
        gnupg \
        git \
        ninja-build \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
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

# venv with python 3.8 (INSTALL.md spec)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv ${UV_PROJECT_ENVIRONMENT} --python 3.8

# torch 1.9.1 + cu111 (INSTALL.md)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${UV_PROJECT_ENVIRONMENT}/bin/python \
        torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
        -f https://download.pytorch.org/whl/torch_stable.html

# build deps for mmcv / mmdet3d. setuptools 59.5.0 (upstream Dockerfile pin — 신 setup.py 호환).
# scikit-image 0.19.3 사용 (0.20.0 은 numpy>=1.21.1 요구하나 numpy 는 requirements.txt 와 맞춰
# 1.20.0 으로 고정해야 함). upstream Dockerfile 은 pip 의 느슨한 처리로 우회했지만 uv 는 strict.
# Pillow<10 — 10.x 부터 PIL._typing 이 numpy.typing.NDArray 호출하지만 numpy 1.20 에 부재.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${UV_PROJECT_ENVIRONMENT}/bin/python \
        numpy==1.20.0 \
        scipy==1.7.3 \
        scikit-image==0.19.3 \
        setuptools==59.5.0 \
        wheel \
        Cython \
        "Pillow<10"

# mmcv-full 1.4.0 (prebuilt cu111/torch1.9.0)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${UV_PROJECT_ENVIRONMENT}/bin/python \
        --no-build-isolation \
        mmcv-full==1.4.0 \
        -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# mmdet + mmsegmentation
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${UV_PROJECT_ENVIRONMENT}/bin/python \
        --no-build-isolation \
        mmdet==2.14.0 \
        mmsegmentation==0.14.1

# mmdet3d v0.17.1 from source (no prebuilt — INSTALL.md 의 git clone + checkout 절차).
# non-editable install 사용 — setuptools 59.5.0 은 PEP 660 의 build_editable hook 부재라
# uv 의 -e 가 실패. upstream Dockerfile 도 editable + cp 로 우회했음. non-editable 은
# venv site-packages 로 복사 (동일 효과).
RUN git clone --branch v0.17.1 --single-branch --depth 1 \
        https://github.com/open-mmlab/mmdetection3d.git /opt/mmdetection3d
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${UV_PROJECT_ENVIRONMENT}/bin/python \
        --no-build-isolation \
        -v /opt/mmdetection3d

# UniAD requirements + torchmetrics 0.8.2 pin (upstream Dockerfile 주석: "old version needed")
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${UV_PROJECT_ENVIRONMENT}/bin/python \
        --no-build-isolation \
        -r /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${UV_PROJECT_ENVIRONMENT}/bin/python \
        --no-build-isolation \
        torchmetrics==0.8.2

# ssh session 도 venv PATH 받도록 (gameformer KAK-17 패턴):
# Docker ENV 는 PID 1 만 적용 — sshd fork session 은 안 받음.
# - /etc/environment: PAM 이 모든 session 시작 시 load
# - /etc/bash.bashrc: interactive bash 추가 보장
RUN printf 'PATH="/opt/venv/uniad/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"\n' > /etc/environment \
    && printf '\nexport PATH=/opt/venv/uniad/bin:$PATH\n' >> /etc/bash.bashrc

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["sleep", "infinity"]
