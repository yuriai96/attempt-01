FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# CUDA arch list for torch ( 8.0 A100 - 8.6 A40, A6000 -  8.9 6000 ADA - 12.0 6000 PRO BW)
ENV TORCH_CUDA_ARCH_LIST="9.0"

RUN apt update -y &&  \
    apt-get install software-properties-common -y &&  \
    apt update -y &&  \
    add-apt-repository ppa:deadsnakes/ppa -y &&  \
    apt update -y  &&\
    apt install -y wget gnupg

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb

# Install basic system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    git \
    build-essential \
    cmake \
    ninja-build \
    python3.11  \
    python3-pip  \
    python3-wheel \
    libjpeg-dev \
    libpng-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    bash \
    cuda-nvcc-12-8 \
    cuda-libraries-dev-12-8 \
    cuda-nvtx-12-8

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 120 &&  \
update-alternatives --install /usr/bin/python python /usr/bin/python3 120

# Install requirements.txt
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install uv
RUN uv pip install -r /tmp/requirements.txt --system


# Install Flash Attention
RUN uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl --system

# Install SPZ
RUN wget https://github.com/404-Repo/spz/releases/download/v.0.1.2/pyspz-wheels-ubuntu-latest.zip && \
    unzip pyspz-wheels-ubuntu-latest.zip && \
    uv pip install pyspz-1.0.0-cp311-cp311-manylinux_2_34_x86_64.whl --system && \
    rm -rf pyspz-wheels-ubuntu-latest.zip *.whl

# Install ben2
RUN pip install -e "git+https://github.com/PramaLLC/BEN2.git#egg=ben2"
RUN uv pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3 --system

# Build Judge service
RUN python3.11 -m venv /opt/vllm-env
RUN /opt/vllm-env/bin/pip install vllm==0.10.2 --extra-index-url https://download.pytorch.org/whl/cu128  

# Clean
RUN rm -rf /var/cache/* /tmp/* /var/tmp/* ~/.cache /var/lib/apt/lists/*

COPY pipeline_service /workspace

EXPOSE 10006
COPY start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh
WORKDIR /workspace

CMD ["bash", "start.sh"]
