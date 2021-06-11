from pathlib import Path
import torch

def train_model():
    project_dir = Path(__file__).resolve().parents[2]
    train_set_path = str(project_dir) + '/data/processed/training.pt'
    labels_path = str(project_dir) + '/data/processed/labels.pt'
    train_imgs, train_labels = torch.load(train_set_path) # img, label
    labels_as_string = torch.load(labels_path) # img, label
    print(train_imgs[0])
    print(labels_path)
    print(labels_as_string)

    train_set = torch.utils.data.TensorDataset(train_imgs, train_labels)