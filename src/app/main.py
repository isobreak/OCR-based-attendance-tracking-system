import json
from typing import Annotated
import asyncio
from io import BytesIO

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, Form, HTTPException

from celery_app import predict


app = FastAPI()


@app.post("/processing", status_code=202)
async def create_task(files: list[UploadFile], acceptable_names: Annotated[str, Form()]):
    async def get_image(file: UploadFile):
        try:
            contents = await file.read()
            image = Image.open(BytesIO(contents))
            return np.array(image)

        except Exception as e:
            print(f"Error processing file {file.filename}: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {file.filename}")

    images = await asyncio.gather(*(get_image(file) for file in files))
    task = predict.delay(images, json.loads(acceptable_names))

    return {"task_id": task.id}



@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    result = predict.AsyncResult(task_id)

    return {
        "ready": result.ready(),
        "successful": result.successful(),
        "result": result.result if result.ready() else None,
    }
