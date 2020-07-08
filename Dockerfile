from tensorflow/tensorflow:2.2.0-gpu
ENV DEBIAN_FRONTEND noninteractive
COPY ./requirements_docker.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements_docker.txt
RUN apt-get update && apt-get -y install python3-tk

COPY ./model /app/model
COPY ./preprocessing /app/preprocessing
COPY ./postprocessing /app/postprocessing
COPY ./run_training.py /app/run_training.py
COPY ./generate_fakes.py /app/generate_fakes.py
COPY ./preprocess_data.py /app/preprocess_data.py
COPY ./generate_fakes.py /app/generate_fakes.py

WORKDIR /app/
