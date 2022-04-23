FROM nvidia/cuda:11.5.0-cudnn8-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y python3 python3-pip sudo

RUN useradd -m zihadul

RUN chown -R zihadul:zihadul /home/zihadul/

COPY --chown=zihadul . /home/zihadul/app

USER zihadul

# RUN cd /home/zihadul/app/ && pip3 install -r requirements.txt

WORKDIR /home/zihadul/app