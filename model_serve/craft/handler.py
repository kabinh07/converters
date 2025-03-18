from pathlib import Path

from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import PredictionException
import torch
import torch.backends.cudnn as cudnn
import os
from PIL import Image
from io import BytesIO
import base64
import numpy as np
from craft_utils import diff, group_text_box
from detector import get_textbox
import cv2


class ModelHandler(BaseHandler):
    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None
        self.manifest = None

    def initialize(self, context):
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get('model_dir')
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        serialized_file = self.manifest['model']['serializedFile']
        model_pth_path = Path(str(os.path.join("models", "CRAFT_clr_best_jit.pth")))
        self.model = torch.jit.load(str(model_pth_path), map_location=self.device)
        self.model.to(self.device)
        cudnn.benchmark = False
        self.model.eval()
        self.initialized = True

    def preprocess(self, data):
        data = data[0].get("body") or data[0].get("data")
        img = data.get("img")
        if not img:
            raise PredictionException("Invalid data provided for inference", 513)
        split = img.strip().split(',')
        if len(split) < 2:
            raise PredictionException("Invalid image", 513)
        img = np.array(Image.open(BytesIO(base64.b64decode(split[1]))).convert("RGB"))
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        binary = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        image = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(image)
        return image

    def box_predict(self, img, min_size = 25, text_threshold = 0.1, low_text = 0.35,\
               link_threshold = 0.5,canvas_size = 768, mag_ratio = 1,\
               slope_ths = 0.2, ycenter_ths = 1.0, height_ths = 0.5,\
               width_ths = 1.0, add_margin = 0.3, reformat=True, optimal_num_chars=None,
               threshold = 0.2, bbox_min_score = 0.2, bbox_min_size = 5, max_candidates = 0,):
        text_box_list = get_textbox(self.model,
                                    img,
                                    canvas_size=canvas_size,
                                    mag_ratio=mag_ratio,
                                    text_threshold=text_threshold,
                                    link_threshold=link_threshold,
                                    low_text=low_text,
                                    poly=False,
                                    device=self.device,
                                    optimal_num_chars=optimal_num_chars,
                                    threshold=threshold,
                                    bbox_min_score=bbox_min_score,
                                    bbox_min_size=bbox_min_size,
                                    max_candidates=max_candidates,
                                    )
        horizontal_list_agg, free_list_agg = [], []
        for text_box in text_box_list:
            horizontal_list, free_list = group_text_box(text_box, slope_ths,
                                                        ycenter_ths, height_ths,
                                                        width_ths, add_margin,
                                                        (optimal_num_chars is None))
            if min_size:
                horizontal_list = [i for i in horizontal_list if max(
                    i[1] - i[0], i[3] - i[2]) > min_size]
                free_list = [i for i in free_list if max(
                    diff([c[0] for c in i]), diff([c[1] for c in i])) > min_size]
            horizontal_list = [self.__add_padding(box, 0, 0) for box in horizontal_list]
            free_list = [self.__add_padding(box, 0, 0) for box in free_list]
            horizontal_list_agg.append(horizontal_list)
            free_list_agg.append(free_list)

        return horizontal_list_agg, free_list_agg

    def inference(self, model_input):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediction output
        :param model_input: Input data for inference
        :return:   output
        """
        horizontal_list, free_list = self.box_predict(img=model_input)
        horizontal_list, free_list = horizontal_list[0], free_list[0]
        hl = [np.array(item).tolist() for item in horizontal_list]
        fl = [np.array(item).tolist() for item in free_list]
        return [{"horizontal_list": hl, "free_list": fl}]

    def postprocess(self, model_output):
        return [model_output]

    def __add_padding(self, box, x_pad, y_pad):
        box[0] -= x_pad
        box[1] += x_pad
        box[2] -= y_pad
        box[3] += y_pad
        return box
