import json
import random
from pathlib import Path

import kornia
import matplotlib.pyplot as plt
import torch


def check_data():
    project_dir = Path(__file__).resolve().parents[2]
    train_set_path = str(project_dir) + "/data/processed/training.pt"
    mapping_file_path = str(project_dir) + "/data/processed/mapping.json"
    train_imgs, train_labels = torch.load(train_set_path)  # img, label

    with open(mapping_file_path) as json_file:
        mapping = json.load(json_file)
    print("All fish types: ", mapping)

    randomInt = random.randint(0, train_imgs.shape[0] - 1)

    print("length of dataset: ", len(train_imgs))
    print("Showing fish: ", mapping[str(int(train_labels[randomInt].item()))])
    plt.imshow(
        kornia.utils.tensor_to_image(
            train_imgs[randomInt]
        )
    )
    plt.show()

    # train_set = torch.utils.data.TensorDataset(train_imgs, train_labels)
