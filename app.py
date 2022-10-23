from fastapi import FastAPI, File
from model import get_model, get_prediction
from image_transformation import get_image
from starlette.responses import Response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title = "Vision-based object detection API",
    description = """ Vision-based vehicle and people detection. 
    Response is in JSON format or in image formato with labels""",
    version = "1.0.0",
)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ['*'],
    allow_headers = ['*'],
)

# Load You Only Look Once V5 medium model
model = get_model()

@app.post("/json-output")
async def object_detection(file: bytes = File(...)):
    image_input = get_image(file)
    response_json = get_prediction(model, image_input, 'json')
    return{'object_detection_result': response_json}

@app.post("/image-output")
async def object_detection(file: bytes = File(...)):
    image_input = get_image(file)
    response_image, image_bytes = get_prediction(model, image_input, 'image')
    return Response(content = image_bytes.getvalue(), headers = response_image, media_type = 'image/jpeg')


