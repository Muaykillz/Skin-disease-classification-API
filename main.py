from fastapi import FastAPI, UploadFile, File
from fastai.vision.all import *
from fastai.vision.core import PILImage
from PIL import Image
import io
import numpy as np
import pathlib
plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

app = FastAPI()

model = models.mobilenet_v2
model = get_model(model)

def get_model():
    
    def _mobilenet_v2_split(m:nn.Module): return L(m[0][0][:7],m[0][0][7:], m[1:]).map(params)
    _mobilenet_v2_meta   = {'cut':-1, 'split':_mobilenet_v2_split, 'stats':imagenet_stats}
    model_meta[models.mobilenet_v2] = {**_mobilenet_v2_meta}
    model = load_learner('./model_1.pkl')
    
    return model

@app.get("/")
async def root():
    return {"message": "Hello World!"}

@app.post("/predict")
async def image_cls(image: UploadFile = File(...)):
    
    image_data = await image.read()
    img = Image.open(io.BytesIO(image_data))
    img_array = np.array(img)
    print(img_array)

    img_array = PILImage.create(img_array)
    result = model.predict(img_array)
    print(result)

    return {"label": f"{result[0]}", "proba": str(result[2].max().numpy())}