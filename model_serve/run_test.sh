#!/bin/bash

docker run -it --gpus all --rm \
    -v ./test.py:/app/test.py \
    -v ./craft/handler.py:/app/craft/handler.py \
    -v ./tmp:/app/tmp \
    mar_test:latest