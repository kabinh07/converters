#!/bin/bash

docker run -it --gpus all --rm \
    -v ./test.py:/app/test.py \
    -v ./craft/handler.py:/app/craft/handler.py \
    -v ./facenet/handler.py:/app/facenet/handler.py \
    -v ./tmp:/app/tmp \
    -v ./models:/models \
    mar_test:latest