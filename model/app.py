from fastapi import FastAPI, File, UploadFile
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torchvision import transforms
import numpy as np
import rasterio
from PIL import Image

app = FastAPI()

class WaterSurfaceSegmentation:
    def __init__(self, model_path):
        self.model = self.load_unet_model(model_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 10, std=[0.5] * 10)
        ])

    def load_unet_model(self, path_to_state_dict):
        model = self.create_unet_with_10_channels()
        model.load_state_dict(torch.load(path_to_state_dict, map_location=torch.device('cpu')))
        model.eval()
        return model

    def create_unet_with_10_channels(self):
        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=10,
            classes=1
        )

    def predict(self, image_array):
        image_tensor = self.transform(image_array).unsqueeze(0)
        with torch.no_grad():
            prediction = self.model(image_tensor)
        return prediction.squeeze().numpy()
    
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with rasterio.open(file.file) as src:
        image = src.read()
    if image.shape[0] != 10:
        return {"error": "Image must have 10 channels."}
    img_tensor = torch.from_numpy(image).float()
    img_tensor = torch.clamp(img_tensor, min=0, max=5000)
    img_tensor = img_tensor / 5000

    model_path = "best_model_dice.pth"
    segmenter = WaterSurfaceSegmentation(model_path)
    mask = segmenter.predict(image)
    
    return {"mask": mask.tolist()}