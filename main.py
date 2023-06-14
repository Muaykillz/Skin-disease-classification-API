from fastapi import FastAPI, UploadFile, File
# from fastai.vision.all import *
# from fastai.vision.core import PILImage
# from PIL import Image
# import io
# import numpy as np
# import pathlib
# plt = platform.system()
# if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World!"}

@app.post("/predict")
async def image_cls(image: UploadFile = File(...)):
    model = load_learner('./model1.pkl')
    image_data = await image.read()
    img = Image.open(io.BytesIO(image_data))
    img_array = np.array(img)
    print(img_array)

    img_array = PILImage.create(img_array)
    result = model.predict(img_array)
    print(result)

    return {"label": f"{result[0]}", "proba": str(result[2].max().numpy())}