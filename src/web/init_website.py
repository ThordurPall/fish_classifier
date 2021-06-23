from __future__ import unicode_literals

import json
import os
# import time
from pathlib import Path

import kornia
import kornia.augmentation as K
import matplotlib.pyplot as plt
# import numpy as np
import torch
import torchvision.transforms as transforms
from bottle import Bottle, request  # , response, route, static_file
from PIL import Image
from resizeimage import resizeimage

from src.models.Classifier import Classifier

project_dir = Path(__file__).resolve().parents[2]
mapping_file_path = str(project_dir) + "/data/processed/mapping.json"

mapping = {}
with open(mapping_file_path) as json_file:
    mapping = json.load(json_file)

# Hyper parameters
batch_size = 64
num_classes = len(mapping)
rgb = 3
height = 64
width = 64
filter1_in = rgb
filter1_out = 6
kernel = 2
pool = 2
filter2_out = 16
filter3_out = 48
fc_1 = 120
fc_2 = 84
pad = 0
stride = 1
lr = 0.001
epochs = 30

model = Classifier(
    num_classes,
    filter1_in,
    filter1_out,
    filter2_out,
    filter3_out,
    height,
    width,
    pad,
    stride,
    kernel,
    pool,
    fc_1,
    fc_2,
)
model.load_state_dict(torch.load((str(project_dir) + "/models/trained_model.pth")))

app = Bottle()


template = """<html>
<head><title>Home</title></head>
<body>
<h1>Upload a file</h1>
<form method="POST" enctype="multipart/form-data" action="/">
<input type="file" name="uploadfile" /><br>
<input type="submit" value="Submit" />
</form>
</body>
</html>"""


@app.get("/")
def home():
    return template


@app.post("/")
def upload():
    # A file-like object open for reading.
    upload_file = request.POST["uploadfile"]
    # Your analyse_data function takes a file-like object and returns a new
    # file-like object ready for reading.
    # Return a file-like object.

    project_dir = Path(__file__).resolve().parents[2]
    # name, ext = os.path.splitext(upload.filename)
    # if ext not in ('.png', '.jpg', '.jpeg'):
    #    return "File extension not allowed."

    name, ext = os.path.splitext(upload_file.filename)

    file_path = str(project_dir) + "/src/web/tmp/" + name + ext
    upload_file.save(file_path, overwrite=True)

    image = Image.open(file_path)

    image = resizeimage.resize_contain(
        image, [64, 64], resample=Image.LANCZOS, bg_color=(0, 0, 0, 0)
    )

    image = image.convert("RGB")

    image_size_x, image_size_y = 64, 64

    # Perform some transformations on the image
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((image_size_x, image_size_y)),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    image = transform(image)

    # Perform kornia operations on the image
    aff = K.RandomAffine(360, return_transform=True, same_on_batch=True)
    cj = K.ColorJitter(0.2, 0.3, 0.2, 0.3)
    img_out, _ = aff(cj(image))

    processed_image_src = str(project_dir) + "/src/web/tmp/" + name + "_processed.jpg"

    plt.imshow(kornia.utils.tensor_to_image(img_out))
    plt.savefig(processed_image_src)

    result = model(img_out).tolist()[0]
    index = result.index(max(result))

    return """<html>
        <head><title>Fish Quesser</title></head>
        <body>
        <p>This fish is a {fish}</p>
        </form>
        </body>
        </html>""".format(
        fish=mapping[str(index)]
    )


if __name__ == "__main__":
    app.run(debug=True)
