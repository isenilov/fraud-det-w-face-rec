FROM python:3.6
COPY . /app
WORKDIR /app
RUN apt-get update && apt-get install -y \
    ffmpeg && \
    pip install -r requirements.txt --no-cache-dir
CMD python ./main.py