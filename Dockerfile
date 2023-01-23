FROM python:3.10
# FROM tensorflow/tensorflow:2.9.2
FROM nvidia/cuda:11.7.0-base-ubuntu22.04



RUN apt-get update -y
RUN apt-get install -y vim
RUN apt-get install -y pip

RUN apt-get install -y curl
COPY nvidia-container-runtime-script.sh /tmp/nvcontainer.sh
RUN /bin/bash /tmp/nvcontainer.sh
RUN apt-get install -y nvidia-cuda-toolkit
RUN apt-get install -y nvidia-docker2
RUN apt-get install -y nvidia-container-toolkit
RUN apt-get install -y libcudnn8 libcudnn8-dev

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt
RUN rm -f /tmp/requirements.txt

RUN  mkdir /app
COPY src /app/src
COPY data /app/data
COPY models /app/models
RUN ls /app

ENV PATH=/root/.local:$PATH
ENV PATH=/usr/lib/cuda:/usr/local/cuda-11.7/lib64:$PATH

# CMD ["python", "reefscan_inference.py"]