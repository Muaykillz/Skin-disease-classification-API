from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import numpy as np
import pathlib
from fastai.vision.all import *
from torchvision import transforms
plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

app = FastAPI()

model_path = './model_2.pth'
model = torch.load(model_path, map_location=torch.device('cpu'))

@app.get("/")
async def root():
    return {"message": "Hello World!"}

@app.post("/predict")
async def image_cls(image: UploadFile = File(...)):
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class_names = ['AGEP', 'Angioedema', 'DRESS', 'FDE', 'SJSTEN', 'Urticaria']

    image_data = await image.read()
    img = Image.open(io.BytesIO(image_data))
    image_data = None
    img = transform(img).unsqueeze(0)
    transform = None
    # print(img.shape)
    # print(img.dtype)

    with torch.no_grad():
        output = model(img)

    img = None

    probabilities = torch.softmax(output, dim=1)[0]
    _, predicted_class = torch.max(probabilities, dim=0)
    predicted_label = class_names[predicted_class]
    predicted_probability = probabilities[predicted_class].item()
    class_names = None

    return {"label": f"{predicted_label}", "proba": predicted_probability}