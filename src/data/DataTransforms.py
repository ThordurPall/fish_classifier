import numpy as np
import torchvision.transforms as transforms

from src.models.Hyperparameters import Hyperparameters as hp


class DataTransforms:
    def __init__(self):
        self.config = hp().config

    def PIL_image_to_tensor(self, image):
        image = image.convert("RGB")
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (self.config["image_size_x"], self.config["image_size_y"])
                ),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        return transform(image)

    def tensor_to_flat_numpy_array(self, tensor):
        return np.array(tensor).ravel()
