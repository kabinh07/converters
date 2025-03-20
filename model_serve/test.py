import sys

sys.path.append("/app/craft")

from facenet.handler import FaceNetTRTHandler
from craft.handler import ModelHandler
import base64
from io import BytesIO
import cv2
import torch
from PIL import Image, ImageDraw
import numpy as np
import os

def draw_and_save(data, model_output):
    data = data[0].get("body") or data[0].get("data")
    result = model_output[0].get("horizontal_list")
    img = data.get("img")
    if not img:
        raise PredictionException("Invalid data provided for inference", 513)
    split = img.strip().split(',')
    if len(split) < 2:
        raise PredictionException("Invalid image", 513)
    img = Image.open(BytesIO(base64.b64decode(split[1]))).convert("RGB")
    draw = ImageDraw.Draw(img)
    for box in result:
        box = (box[0], box[2], box[1], box[3])
        draw.rectangle(box, outline="#008000", width=3)
    files = os.listdir("tmp")
    img.save(f"tmp/image_{len(files)}.jpg")
    return


def image_to_base64(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    
    # Encode the image to a memory buffer
    _, buffer = cv2.imencode('.jpg', image)
    
    # Convert the buffer to a base64 string
    img_str = base64.b64encode(buffer).decode('utf-8')
    
    # Return the image as a base64 string
    return f"data:image/jpeg;base64,{img_str}"

def test_facenet():
    model_handler = FaceNetTRTHandler()
    model_handler.initialize(None)  # Simulate initialization with no context

    image_1_base64 = image_to_base64('image_1.jpg')  # Replace with your image path
    image_2_base64 = image_to_base64('image_2.jpg')  # Replace with your image path

    # Prepare the data in the expected format (list of dictionaries with image data)
    data = [
        {
            "body": {
                "img_1": image_1_base64,
                "img_2": image_2_base64
            },
        },
    ]

    context = {
        "system_properties": {
            "model_dir": "/models"
        }
    }

    # Test inference
    with torch.no_grad():
        result = model_handler.handle(data, context)
    
    print("Inference result:", result)

def test_craft():
    model_handler = ModelHandler()
    model_handler.initialize(None)  # Simulate initialization with no context

    image_base64 = image_to_base64('address_first_cropped_img.png')  # Replace with your image path

    # Prepare the data in the expected format (list of dictionaries with image data)
    data = [
        {
            "body": {
                "img": image_base64
            },
        },
    ]
    # print(f"Data: {data}")
    img = model_handler.preprocess(data)

    # Test inference
    with torch.no_grad():
        result = model_handler.inference(img)
    
    print("Inference result:", result)
    draw_and_save(data, result)
    sys.exit()

if __name__ == "__main__":
    test_facenet()