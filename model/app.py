from fastapi import FastAPI, File, UploadFile
import torch
import rasterio

from model import WaterSurfaceSegmentation


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'{DEVICE= }')

app = FastAPI()
    
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with rasterio.open(file.file) as src:
        image = src.read()
    if image.shape[0] != 10:
        return {"error": "Image must have 10 channels."}
    
    img_tensor = torch.from_numpy(image).float()
    img_tensor = torch.clamp(img_tensor, min=0, max=5000)
    img_tensor = img_tensor / 5000

    # Загрузка модели и предсказание
    model_path = "best_model.pth"
    segmenter = WaterSurfaceSegmentation(model_path)
    mask = segmenter.predict(img_tensor)
    print(mask.shape)
    return {"mask": mask.tolist()}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)