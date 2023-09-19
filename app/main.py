from model import predict_

from fastapi import FastAPI, UploadFile
import io
from PIL import Image

app = FastAPI()


# @app.get("/")
# def read_root():
#     return {"Hello": "World"}


@app.post("/predict")
def neural_predict(image: UploadFile):
    content = image.file.read()
    image_bytes = io.BytesIO(content)

    result = predict_(image_bytes)
    return result
