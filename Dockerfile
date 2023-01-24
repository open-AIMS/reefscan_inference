FROM nvidia/cuda:11.7.0-base-ubuntu22.04

RUN apt-get update -y
RUN apt-get install -y vim
RUN apt-get install -y pip

RUN apt-get install -y nvidia-cuda-toolkit
RUN apt-get install -y libcudnn8 libcudnn8-dev

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt
RUN rm -f /tmp/requirements.txt

RUN  mkdir /app
COPY src /app/src
COPY data /app/data
COPY models /app/models
COPY results /app/results
RUN ls /app

RUN rm -f /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so.1

ENV PATH=/root/.local:$PATH
ENV PATH=/usr/lib/cuda:/usr/local/cuda-11.7/lib64:$PATH

# CMD ["python", "reefscan_inference.py"]