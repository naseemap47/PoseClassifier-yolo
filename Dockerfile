FROM ubuntu:20.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    git
RUN sudo apt-get remove -y '^ghc-8.*'
RUN apt-get remove -y '^dotnet-.*'
RUN apt-get remove -y '^llvm-.*'
RUN apt-get remove -y 'php.*'
RUN apt-get remove -y azure-cli google-cloud-sdk hhvm google-chrome-stable firefox powershell mono-devel
RUN apt-get autoremove -y
RUN apt-get clean
RUN rm -rf /usr/share/dotnet/
RUN git clone https://github.com/naseemap47/PoseClassifier-yolo.git home
WORKDIR /home
RUN pip install -r requirements.txt