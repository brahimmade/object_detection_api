import torch
import json
from image_transformation import get_image_output_transformation

def get_model():
    # load model YOLOV5 medium
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
    model.classes = [0, 2]  # people and car
    model.conf = 0.5 # set model confidence to 50%
    return model

def get_prediction(model, image_input, response_type):
    if response_type == 'json': # For response in JSON format
        results = model(image_input)
        response_json = results.pandas().xyxy[0].to_json(orient = 'records')  # Predictions JSON format
        response_json = json.loads(response_json)
        return response_json

    else: # response in image JPEG format with labels rendered
        results = model(image_input)
        results.render()
        response_image = get_image_output_transformation(results)
        return response_image
