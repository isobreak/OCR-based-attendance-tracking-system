import os

# models
DET_MODEL_PATH = os.environ.get("DET_MODEL_PATH")
REC_MODEL_PATH = os.environ.get("REC_MODEL_PATH")

# hardware requirements
MIN_CUDA_MEMORY_REQUIRED = int(os.environ.get("MIN_CUDA_MEMORY_REQUIRED"))

# urls
BROKER_URL = os.environ.get("BROKER_URL")
BACKEND_URL = os.environ.get("BACKEND_URL")
