from fastai.vision.all import *
import torchvision.models as models
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from PIL import Image

model = models.mobilenet_v2
model = get_model(model)

def get_model():
    
    def _mobilenet_v2_split(m:nn.Module): return L(m[0][0][:7],m[0][0][7:], m[1:]).map(params)
    _mobilenet_v2_meta   = {'cut':-1, 'split':_mobilenet_v2_split, 'stats':imagenet_stats}
    model_meta[models.mobilenet_v2] = {**_mobilenet_v2_meta}
    model = load_learner('./model_1.pkl')
    
    return model
    

