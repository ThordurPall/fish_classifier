import base64
from io import BytesIO

import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from src.models.Hyperparameters import Hyperparameters as hp


class DataTransforms:
    def __init__(self):
        self.config = hp().config

    def PIL_image_to_tensor(self, image):
        image = image.convert("RGB")
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.config["image_height"], self.config["image_width"])
                ),
                transforms.ToTensor(),
            ]
        )
        return transform(image)

    def tensor_to_flat_numpy_array(self, tensor):
        return np.array(tensor).ravel()

    def PIL_image_to_b64(self, image, utf8=True):
        image = image.convert("RGB")
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        if utf8:
            return img_str.decode("utf-8")
        return img_str

    def b64_to_PIL_image(self, image):
        return Image.open(BytesIO(base64.b64decode(image)))
