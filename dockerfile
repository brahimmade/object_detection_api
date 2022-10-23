FROM tiangolo/uvicorn-gunicorn:python3.9-slim

LABEL maintainer="pipe11"

ENV WORKERS_PER_CORE=3
ENV MAX_WORKERS=24
ENV LOG_LEVEL="warning"
ENV TIMEOUT="200"

RUN mkdir /object_detection_api

COPY requirements.txt /object_detection_api

COPY . /object_detection_api

WORKDIR /object_detection_api

RUN pip install -r requirements.txt

EXPOSE 8000

HEALTHCHECK CMD curl --fail http://localhost:8000/ || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]