import torch
import json
import numpy as np
import cv2
import base64
from PIL import Image
from io import BytesIO
from ts.torch_handler.base_handler import BaseHandler
from scipy.spatial.distance import cosine

class FaceNetTRTHandler(BaseHandler):
    def __init__(self):
        super(FaceNetTRTHandler, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def initialize(self, ctx):
        """Load the TensorRT model and initialize inference context."""
        # properties = ctx.system_properties
        model_path = f"models/facenet.pt"
        torch_model = torch.jit.load(model_path)
        self.model = torch.compile(torch_model, backend = "tensorrt")
        self.model = self.model.to(self.device)

    def preprocess(self, data):
        """Preprocess images using OpenCV for face detection and TensorRT for face recognition."""
        preprocessed_images = []
        for row in data:
            image = row.get("data") or row.get("body")
            if isinstance(image, list):  # If the image data is a list, get the first element
                image = image[0]
            split = image.strip().split(',')
            if len(split) < 2:
                raise PredictionException("Invalid image", 513)
            img_data = base64.b64decode(split[1])
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise PredictionException("Failed to decode image", 513)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (160, 160))

            # Detect faces using OpenCV (Haar Cascade)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) == 0:
                continue  # No faces detected
            
            for (x, y, w, h) in faces:
                face = image[y:y+h, x:x+w]  # Crop the face from the image
                face = cv2.resize(face, (160, 160))  # Resize to 160x160 for FaceNet
                face = face.astype(np.float32) / 255.0  # Normalize
                face = np.transpose(face, (2, 0, 1))  # HWC to CHW
                face = np.expand_dims(face, axis=0)  # Add batch dimension
                # face = torch.tensor(face, dtype = torch.float32)
                # face = face.to(self.device)
                preprocessed_images.append(face)

        if not preprocessed_images:
            return None
        
        return np.vstack(preprocessed_images)

    def inference(self, data):
        """Run inference."""
        if data is None:
            return None
        output_tensor = []
        for image in data:
            with torch.no_grad():
                # Make sure the image is on the same device
                image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).permute(0, 2, 3, 1).to(self.device)
                result = self.model(image)  # Inference on the image
                output_tensor.append(result.cpu().numpy())  # Collect results as numpy arrays
        return output_tensor

    def calculate_similarity(self, emb1, emb2):
        """Calculate the cosine similarity between two embeddings."""
        return 1 - cosine(emb1[0], emb2[0])

    def postprocess(self, inference_output):
        """Postprocess results and compute similarity."""
        if inference_output is None or len(inference_output) < 2:
            return [{"error": "Insufficient faces detected"}]
        
        emb1, emb2 = inference_output[0], inference_output[1]
        similarity = self.calculate_similarity(emb1, emb2)
        
        return [{"similarity": similarity}]

    def handle(self, data, context):
        """Main handler function for TorchServe."""
        # Preprocess both images
        images = self.preprocess(data)
        
        # Get embeddings for both faces
        embeddings = self.inference(images)
        
        # Compute similarity between the two embeddings
        return self.postprocess(embeddings)
