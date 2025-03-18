#!/bin/bash

docker run \
    --gpus all \
    -it --rm \
    -v ./main.py:/workspace/main.py \
    nvcr.io/nvidia/tensorrt:24.01-py3 \
    bash -c \
    "nvidia-smi"