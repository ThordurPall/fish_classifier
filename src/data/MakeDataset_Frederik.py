import gdown
import zipfile
import os.path
import kornia
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from pathlib import Path
import kornia.augmentation as K
import torch.nn as nn
import numpy as np
from PIL import Image


class MakeDataset():
    """
    A class that handles everything related to getting
    and setting up the training and test datasets

    ...

    Methods
    -------
    download_data()
        Downloads the data from Google Drive and unzips it
    """


    def __init__(self,
                file_url='https://drive.google.com/uc?id=1WyVALHmHhPdt5e-XLSgFZ8aeg6dSI0dI',
                force_download=False,
                force_unzip=False):
        super().__init__()
        project_dir = Path(__file__).resolve().parents[2]
        self.file_url = file_url
        self.force_download = force_download
        self.force_unzip = force_unzip
        self.raw_zip_folder = str(project_dir) + '/data/raw/unzipped'
        self.raw_zip_file = str(project_dir) + '/data/raw/raw.zip'
        self.processed_files_folder = str(project_dir) + '/data/processed'

    def make_dataset(self):
        self.download_data()
        self.unzip_data()
        self.augment()

    def download_data(self):
            """ Downloads the data from Google Drive """
            # Check if the file already exists
            if not os.path.isfile(self.raw_zip_file) or self.force_download:
                print('Downloading data')
                gdown.download(self.file_url,
                            self.raw_zip_file,
                            quiet=False)
                print('Data successfully downloaded')

    def unzip_data(self):
        """ Unzips the raw data zip file """
        if not os.path.isdir(self.raw_zip_folder) or self.force_unzip:
            print('Unzipping data')
            with zipfile.ZipFile(self.raw_zip_file, 'r') as zip_ref:
                zip_ref.extractall(self.raw_zip_folder)
            print('Data successfully unzipped')

    def augment(self):
        """ Applies data augmentation to generate 1000 images pr image"""
        path = Path(__file__).resolve().parents[2]
        file = path.joinpath(self.raw_zip_folder+'/NA_Fish_Dataset/Black Sea Sprat/00002.png')

        img = Image.open(file).convert('RGB')

        transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((256,256)),
                                transforms.Normalize((0.5,), (0.5,))])
        
        img = transform(img)

        print(type(img))
        img_in = kornia.tensor_to_image(img)  # HxWxC / np.uint8

        self.aff = K.RandomAffine(360, p=1.0, return_transform=True, same_on_batch=True)
        self.cj = K.ColorJitter(0.2, 0.3, 0.2, 0.3)
            
        for _ in range(10):
            img_out, matr = self.aff(self.cj(img))
            print(matr)
            img_out = kornia.tensor_to_image(img_out)  # HxWxC / np.uint8

            _, axs = plt.subplots(1, 2, figsize=(8, 4))
            axs = axs.ravel()

            axs[0].axis('off')
            axs[0].imshow(img_in)
            axs[1].axis('off')
            axs[1].imshow(img_out)

            plt.show()
