from fastai.vision.all import *
import torch
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from PIL import Image

# folder_path = './'
# fname = 'model1.pkl'

# img = 

model = load_learner('model1.pkl')