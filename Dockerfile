# Base image
FROM python:3.8-slim



RUN apt-get update && apt-get install -y python3-opencv

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir vtk

ARG INPUT_DIR
ARG OUTPUT_DIR

COPY requirements.txt requirements.txt

ADD main.py .
CMD ["python", "./main.py"]

