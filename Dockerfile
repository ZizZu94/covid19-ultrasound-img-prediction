FROM nvidia/cuda:11.6.0-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y python3 python3-pip sudo

RUN useradd -m zihadul

RUN chown -R zihadul:zihadul /home/zihadul/

COPY --chown=zihadul . /home/zihadul/app

USER zihadul

RUN cd /home/zihadul/app/ && pip3 install --no-cache-dir -r requirements.txt

WORKDIR /home/zihadul/app