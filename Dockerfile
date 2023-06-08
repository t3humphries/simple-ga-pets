# syntax=docker/dockerfile:1
FROM python:3.8-buster
RUN apt-get install gcc
COPY . /DP_Simple_GA/
WORKDIR /DP_Simple_GA
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN cd pam_algo && python3 setup.py install
RUN unzip processed_datasets.zip



