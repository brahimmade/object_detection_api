# Vision-based object detection API
**Vision-based vehicle and people detection as a web service**. Response from the API is in JSON format or in image format with prediction labels.

## Table of contents


**[1. Introduction](#1.-Introduction)**

**[2. Object detection model](#2.-Object-detection-model)**

**[3. Web service](#3.-Web-service)**

**[4. References](#4.-References)**


## <div align="center">Quick Start methods</div>

<details closed>
<summary>Install and start using local service without Docker</summary>

Clone repo and install [requirements.txt](https://github.com/pipe11/object_detection_api/blob/master/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, then launch the app:

```bash
git clone https://github.com/pipe11/object_detection_api  # clone
cd object_detection_api
pip install -r requirements.txt  # install
uvicorn app:app --host 0.0.0.0 --port 8000 # launch app
```
Finally, go to [http://localhost:8000/docs](http://localhost:8000/docs), check the FastAPI interactive doc and try it out with POST method, selecting output format and loading an image with cars and people:
- **POST json-output**
- **POST image-output**

Collections file: [**collection_object_detection_api_local**](ttps://github.com/pipe11/object_detection_api/blob/master/collection_object_detection_api_local)  for other API frameworks such us [**Postman**](https://www.postman.com/) or [**Thunder Client**](https://marketplace.visualstudio.com/items?itemName=rangav.vscode-thunder-client)

</details>

<details open>
<summary>Install, container deployment in Docker and run the service</summary>

Clone repo and install [**requirements.txt**](https://github.com/pipe11/object_detection_api/blob/master/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, then open [**Docker deskopt**](https://www.docker.com/products/docker-desktop) and finally launch the app:

```bash
git clone https://github.com/pipe11/object_detection_api  # clone
cd object_detection_api
pip install -r requirements.txt  # install
docker build -t object_detection_api:1.0.0 # docker build
```
Run container in detached mode (Background mode):
```bash
docker run -ti -d --name object_detection_api_detached -p 8000:8000 object_detection_api:1.0.0
```
Run container in default mode (Foreground):
```
docker run -ti -d --name object_detection_api_foreground -p 8000:8000 object_detection_api:1.0.0
```
Finally, go to [http://localhost:8000/docs](http://localhost:8000/docs), check the FastAPI interactive doc and try it out with POST method, selecting output format and loading an image with cars and people:
- **POST json-output**
- **POST image-output**

</details>

## 1. Introduction
The purpose of this exercise is the development of a local service (TODO online Service) for image object detection, specifically for car and people detection. This service will have as input an image and will provide an output in JSON format with the detection of cars and people. Additionally, the service provides an extra output in image format with labels of the detections.

Main keys of the project:
- A pre-trained computer vision model specialized in object detection has been used, specifically the **[YOLOv5 model](https://github.com/ultralytics/yolov5)**

- The object detection service has been developed with the **[FastAPI framework](https://fastapi.tiangolo.com/)**

- Finally, the service has been containerized in a prepared docker image that exposes the object detection service at launch.


## 2. Object detection model

The artificial intelligence model used in the development of this object detection service is the YOLOv5 model: **[YOLOv5](https://github.com/ultralytics/yolov5)** is a family of object detection architectures and models pre-trained on the **[COCO dataset](https://cocodataset.org/#home)**

**YOLO**, an acronym for "You only look once", is an object detection algorithm that divides images into a grid network. Each grid cell is responsible for detecting objects within the grid.

YOLO is one of the most famous object detection algorithms for its high speed and precision.

#### Short history of YOLOv5:
Shortly after the release of YOLOv4 **[Glenn Jocher](https://github.com/glenn-jocher)** introduced YOLOv5 using the Pytorch framework. The open source code is available on GitHub. Released: 18 May 2020

#### YOLOv5 model selection:
Se ha seleccionado el modelo mediando debido a su alto rendimiento y eficiencia en tiempos de inferencia del modelo:
<p align="left"><img width="800" src="https://user-images.githubusercontent.com/26833433/155040763-93c22a27-347c-4e3c-847a-8094621d3f4e.png"></p>
<details open>
  <summary>YOLOv5-P5 640 Figure</summary>

#### Pre-trained Checkpoints

| Model                                                                                                | size<br><sup>(pixels) | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Speed<br><sup>CPU b1<br>(ms) | Speed<br><sup>V100 b1<br>(ms) | Speed<br><sup>V100 b32<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>@640 (B) |
|------------------------------------------------------------------------------------------------------|-----------------------|-------------------------|--------------------|------------------------------|-------------------------------|--------------------------------|--------------------|------------------------|
| [YOLOv5n](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n.pt)                   | 640                   | 28.0                    | 45.7               | **45**                       | **6.3**                       | **0.6**                        | **1.9**            | **4.5**                |
| [YOLOv5s](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt)                   | 640                   | 37.4                    | 56.8               | 98                           | 6.4                           | 0.9                            | 7.2                | 16.5                   |
| [YOLOv5m](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5m.pt)                   | 640                   | 45.4                    | 64.1               | 224                          | 8.2                           | 1.7                            | 21.2               | 49.0                   |
| [YOLOv5l](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5l.pt)                   | 640                   | 49.0                    | 67.3               | 430                          | 10.1                          | 2.7                            | 46.5               | 109.1                  |
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5x.pt)                   | 640                   | 50.7                    | 68.9               | 766                          | 12.1                          | 4.8                            | 86.7               | 205.7                  |
|                                                                                                      |                       |                         |                    |                              |                               |                                |                    |                        |
| [YOLOv5n6](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n6.pt)                 | 1280                  | 36.0                    | 54.4               | 153                          | 8.1                           | 2.1                            | 3.2                | 4.6                    |
| [YOLOv5s6](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s6.pt)                 | 1280                  | 44.8                    | 63.7               | 385                          | 8.2                           | 3.6                            | 12.6               | 16.8                   |
| [YOLOv5m6](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5m6.pt)                 | 1280                  | 51.3                    | 69.3               | 887                          | 11.1                          | 6.8                            | 35.7               | 50.0                   |
| [YOLOv5l6](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5l6.pt)                 | 1280                  | 53.7                    | 71.3               | 1784                         | 15.8                          | 10.5                           | 76.8               | 111.4                  |
| [YOLOv5x6](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5x6.pt)<br>+ [TTA](https://github.com/ultralytics/yolov5/issues/303) | 1280<br>1536          | 55.0<br>**55.8**        | 72.7<br>**72.7**   | 3136<br>-                    | 26.2<br>-                     | 19.4<br>-                      | 140.7<br>-         | 209.8<br>-             |

## 3. Web service
This web services was developed with **[FastAPI](https://fastapi.tiangolo.com/)**, a modern and fast (high-performance) web framework for building APIs with Python +3.7. And also with **Torch** library, specifically with the pre-trained model repository **[Torch Hub](https://pytorch.org/hub/)**.

### App structure:
- **[app.py](https://github.com/pipe11/object_detection_api/blob/master/app.py)**: It is used as the main script to launch the FastAPI app, itincludes 2 POST methods depending of the output required: JSON or image labeled.
- **[model.py](https://github.com/pipe11/object_detection_api/blob/master/model.py)**: This script is in charge of loading the model and generating the inference predictions
- **[image_transformation.py](https://github.com/pipe11/object_detection_api/blob/master/image_transformation.py)**: This script is applied to rescale the input image to make it ready for prediction and finally it also includes a function to transform the prediction result into an image with the detection labels.

## 4. References

[Ultralytics repository of YOLOv5 model](https://github.com/ultralytics/yolov5)

[Ultralytics documentation of YOLOv5 model](https://zenodo.org/record/7002879#.Y1bjJHZBxPY)

[FastAPI guide for development](https://fastapi.tiangolo.com/tutorial/first-steps/)

[FastAPI in containers](https://fastapi.tiangolo.com/deployment/docker/)