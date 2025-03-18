FROM pytorch/torchserve:latest-cpu

WORKDIR /app

COPY facenet/requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["python3", "test.py"]