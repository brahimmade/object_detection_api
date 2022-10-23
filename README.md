# Vision-based object detection API
Vision-based vehicle and people detection as a web service. Response from the API is in JSON format or in image format with prediction labels.

## Docker commands
### Docker commands for build

```
docker build -t object_detection_api:1.0.0 .
```

### Docker commands for running

Running container in detached mode (Background mode):

```
docker run -ti -d --name object_detection_api_detached -p 8000:8000 object_detection_api:1.0.0
```

Running container in default mode (Foreground)

```
docker run -ti -d --name object_detection_api_foreground -p 8000:8000 object_detection_api:1.0.0
```