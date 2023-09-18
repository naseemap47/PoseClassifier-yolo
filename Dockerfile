FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
COPY . /home
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6
RUN apt-get autoremove -y
RUN apt-get clean
WORKDIR /home
# RUN pip install tensorflow==2.12.0
# RUN pip install ultralytics
# RUN pip install pandas
# RUN pip install scikit-learn