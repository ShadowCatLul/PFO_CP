import segmentation_models_pytorch as smp
import torch


class WaterSurfaceSegmentation:
    def __init__(self, model_path):
        self.model = self.load_unet_model(model_path)
        self.transform = None
        # transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5] * 10, std=[0.5] * 10)
        # ])

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