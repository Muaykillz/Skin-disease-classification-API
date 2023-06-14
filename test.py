# from fastai.vision.all import *
# import torchvision.models as models
# from torchvision import transforms
# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath
# from PIL import Image

# # model = models.mobilenet_v2

# # def _mobilenet_v2_split(m:nn.Module): return L(m[0][0][:7],m[0][0][7:], m[1:]).map(params)
# # _mobilenet_v2_meta = {'cut':-1, 'split':_mobilenet_v2_split, 'stats':imagenet_stats}
# # model_meta[models.mobilenet_v2] = {**_mobilenet_v2_meta}

# # model = load_learner('./model_1.pkl')

# model_path = './model_2.pth'
# model = torch.load(model_path, map_location=torch.device('cpu'))
# # model.eval()

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# class_names = ['AGEP', 'Angioedema', 'DRESS', 'FDE', 'SJSTEN', 'Urticaria']

# img = Image.open('./test.jpg')
# img_array = np.array(img)
# print(img_array.shape)
# img = transform(img).unsqueeze(0)
# # print(img)
# print(img.shape)
# print(img.dtype)
# # img_array = PILImage.create(img_array)
# # with torch.no_grad():
# #     output = model(img)

# # probabilities = torch.softmax(output, dim=1)[0]
# # _, predicted_class = torch.max(probabilities, dim=0)
# # predicted_label = class_names[predicted_class]
# # predicted_probability = probabilities[predicted_class].item()
# # print(predicted_label, predicted_probability)

