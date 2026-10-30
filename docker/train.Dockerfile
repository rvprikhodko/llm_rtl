# Training and data-preparation environment: CUDA 12.8, PyTorch, transformers/peft, iverilog, yosys.
# The repository is not copied into the image; compose.yaml mounts it at /workspace.
ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
# Empty TORCH_INDEX_URL installs torch from the default (or Nexus) index instead of download.pytorch.org
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_VERSION=
ARG PIP_INDEX_URL_OVERRIDE=
# 1 installs mamba-ssm and causal-conv1d CUDA kernels (needed for Mamba runs, adds build time)
ARG INSTALL_MAMBA=0

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev build-essential ninja-build git curl ca-certificates iverilog yosys \
    && rm -rf /var/lib/apt/lists/*

# Corporate CA certificates (*.crt) placed in docker/certs are trusted by apt, pip and huggingface_hub
COPY docker/certs/ /usr/local/share/ca-certificates/extra/
RUN update-ca-certificates
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    TORCH_CUDA_ARCH_LIST=8.9

RUN set -eu; \
    index="${PIP_INDEX_URL_OVERRIDE:+--index-url $PIP_INDEX_URL_OVERRIDE}"; \
    pip install $index --upgrade pip; \
    torch="torch${TORCH_VERSION:+==$TORCH_VERSION}"; \
    if [ -n "$TORCH_INDEX_URL" ]; then pip install "$torch" --index-url "$TORCH_INDEX_URL"; else pip install $index "$torch"; fi

COPY requirements-train.txt /opt/requirements-train.txt
RUN set -eu; \
    index="${PIP_INDEX_URL_OVERRIDE:+--index-url $PIP_INDEX_URL_OVERRIDE}"; \
    pip install $index -r /opt/requirements-train.txt; \
    if [ "$INSTALL_MAMBA" = "1" ]; then pip install $index --no-build-isolation causal-conv1d mamba-ssm; fi; \
    pip freeze > /opt/requirements-train.lock.txt

WORKDIR /workspace
CMD ["bash"]
