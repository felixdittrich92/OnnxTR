FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ARG SYSTEM=gpu
# NOTE: The following CUDA_VERSION, CUDNN_VERSION, and NVINFER_VERSION are for CUDA 11.8
# - this needs to match exactly with the host system otherwise the onnxruntime-gpu package isn't able to work correct. !!
ARG CUDA_VERSION=11-8
ARG CUDNN_VERSION=8.6.0.163-1
ARG NVINFER_VERSION=8.6.1.6-1

# Enroll NVIDIA GPG public key and install CUDA
RUN if [ "$SYSTEM" = "gpu" ]; then \
    apt-get update && \
    apt-get install -y gnupg ca-certificates wget && \
    # - Install Nvidia repo keys
    # - See: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#network-repo-installation-for-ubuntu
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && apt-get install -y --no-install-recommends \
    # NOTE: The following CUDA_VERSION, CUDNN_VERSION, and NVINFER_VERSION are for CUDA 11.8
    # - this needs to match exactly with the host system otherwise the onnxruntime-gpu package isn't able to work correct. !!
    cuda-command-line-tools-11-8 \
    cuda-cudart-dev-${CUDA_VERSION} \
    cuda-nvcc-${CUDA_VERSION}  \
    cuda-cupti-${CUDA_VERSION}  \
    cuda-nvprune-${CUDA_VERSION}  \
    cuda-libraries-${CUDA_VERSION}  \
    cuda-nvrtc-${CUDA_VERSION}  \
    libcufft-${CUDA_VERSION}  \
    libcurand-${CUDA_VERSION}  \
    libcusolver-${CUDA_VERSION}  \
    libcusparse-${CUDA_VERSION}  \
    libcublas-${CUDA_VERSION}  \
    # - CuDNN: https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#ubuntu-network-installation
    libcudnn${CUDNN_VERSION%%.*}=${CUDNN_VERSION}+cuda${CUDA_VERSION.replace('-', '.')}; \
    libnvinfer-plugin${CUDNN_VERSION%%.*}=${NVINFER_VERSION}+cuda${CUDA_VERSION.replace('-', '.')}; \
    libnvinfer${CUDNN_VERSION%%.*}=${NVINFER_VERSION}+cuda${CUDA_VERSION.replace('-', '.')}; \
fi

RUN apt-get update && apt-get install -y --no-install-recommends \
    # - Other packages
    build-essential \
    pkg-config \
    curl \
    wget \
    software-properties-common \
    unzip \
    git \
    # - Packages to build Python
    tar make gcc zlib1g-dev libffi-dev libssl-dev liblzma-dev libbz2-dev libsqlite3-dev \
    # - Packages for docTR
    libgl1-mesa-dev libsm6 libxext6 libxrender-dev libpangocairo-1.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
fi

# Install Python
ARG PYTHON_VERSION=3.10.13

RUN wget http://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz && \
    tar -zxf Python-$PYTHON_VERSION.tgz && \
    cd Python-$PYTHON_VERSION && \
    mkdir /opt/python/ && \
    ./configure --prefix=/opt/python && \
    make && \
    make install && \
    cd .. && \
    rm Python-$PYTHON_VERSION.tgz && \
    rm -r Python-$PYTHON_VERSION

ENV PATH=/opt/python/bin:$PATH

# Install OnnxTR
ARG ONNXTR_REPO='felixdittrich92/onnxtr'
ARG ONNXTR_VERSION=main
RUN pip3 install -U pip setuptools wheel && \
    pip3 install "onnxtr[$SYSTEM,html]@git+https://github.com/$ONNXTR_REPO.git@$ONNXTR_VERSION"
