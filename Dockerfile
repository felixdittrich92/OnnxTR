ARG BASE_IMAGE

FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ARG SYSTEM
ARG PYTHON_VERSION

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
