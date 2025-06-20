FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
LABEL author="Kirill Ryazantcev <isobreak@mail.com>"

RUN mkdir -p /data/models && mkdir /app

COPY data/models/production/ /data/models/
COPY app_requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src/app/ /app/

WORKDIR /app

ENV DET_MODEL_PATH="../data/models/detection/opt_model.pt"
ENV REC_MODEL_PATH="../data/models/recognition"
ENV MIN_CUDA_MEMORY_REQUIRED=4_000_000_000
ENV C_FORCE_ROOT="true"

EXPOSE 8000

CMD ["celery", "-A", "celery_app", "worker", "--loglevel=INFO"]
