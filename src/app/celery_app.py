import warnings
import numpy as np
import torch
import transformers
from transformers.utils import logging
from celery import Celery
from celery.signals import worker_process_init

from processing import Pipeline, Detector, Recogniser
from config import DET_MODEL_PATH, REC_MODEL_PATH, MIN_CUDA_MEMORY_REQUIRED, BROKER_URL, BACKEND_URL



app = Celery(
    'celery_app',
    broker=BROKER_URL,
    backend=BACKEND_URL,
    worker_concurrency=1,   # due to high GPU-memory consumption
)

app.conf.accept_content = ['json', 'pickle']
app.conf.task_serializer='pickle'
app.conf.result_serializer='pickle'

pipe = None

@app.task
def predict(images: list[np.ndarray], acceptable_names: list[list[str]]) -> dict:
    res = pipe.predict(images, acceptable_names)

    return res

@worker_process_init.connect
def load_pipeline(**kwargs):
    _device = 'cpu'
    if torch.cuda.is_available():
        total_memories = {i: torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())}
        max_idx = max(total_memories, key=total_memories.get)
        if total_memories[max_idx] > MIN_CUDA_MEMORY_REQUIRED:
            _device = f'cuda:{max_idx}'
        else:
            warnings.warn(f'Not enough memory on GPU(s) to load the model. Maximum memory available: '
                          f'{total_memories[max_idx]}, while MIN_CUDA_MEMORY_REQUIRED = {MIN_CUDA_MEMORY_REQUIRED}')
    else:
        warnings.warn(f'CUDA is not available, CPU is used instead', UserWarning)

    logging.set_verbosity(transformers.logging.ERROR)
    _rec = Recogniser(REC_MODEL_PATH, _device)
    print(f'Recognition model "{REC_MODEL_PATH}" is now on "{_device}"')

    _det = Detector(DET_MODEL_PATH, 'cpu')
    print(f'Detection model at "{DET_MODEL_PATH}" is now on cpu')

    global pipe
    pipe = Pipeline(_det, _rec)
