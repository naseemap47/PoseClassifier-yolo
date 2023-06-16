FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    git
RUN apt-get autoremove -y
RUN apt-get clean
RUN git clone https://github.com/naseemap47/PoseClassifier-yolo.git home
WORKDIR /home
RUN pip install -r requirements.txt