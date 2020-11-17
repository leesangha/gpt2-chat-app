FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y \
    python3.7 \
    python3-pip

RUN pip3 install --upgrade pip setuptools

WORKDIR /app
COPY requirements.txt /app
RUN pip3 install -r requirements.txt

COPY . /app

EXPOSE 80
CMD python3 server.py