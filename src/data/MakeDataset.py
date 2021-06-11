import gdown
import zipfile
from pathlib import Path
import os
import matplotlib.pyplot as plt
import kornia.augmentation as K
import torch.nn as nn
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import kornia

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
                 force_unzip=False,
                 force_process=False,
                 training_partition_percentage=0.85):
        super().__init__()
        project_dir = Path(__file__).resolve().parents[2]
        self.file_url = file_url
        self.force_download = force_download
        self.force_unzip = force_unzip
        self.force_process = force_process
        self.data_folder = str(project_dir) + '/data'
        self.raw_unzipped_file_folder = str(project_dir) + '/data/raw/unzipped'
        self.raw_zip_file = str(project_dir) + '/data/raw/raw.zip'
        self.processed_training_set = str(project_dir) + '/data/processed/training.pt'
        self.processed_test_set = str(project_dir) + '/data/processed/test.pt'
        self.processed_files_folder = str(project_dir) + '/data/processed'
        self.processed_labels = str(project_dir) + '/data/processed/labels.pt'
        self.raw_files_folder = str(project_dir) + '/data/raw'
        self.training_partition_percentage = training_partition_percentage

        # Make folders if they do not exist
        if not os.path.isdir(self.data_folder):
            os.mkdir(self.data_folder)
        if not os.path.isdir(self.raw_files_folder):
            os.mkdir(self.raw_files_folder)
        if not os.path.isdir(self.processed_files_folder):
            os.mkdir(self.processed_files_folder)

    def make_dataset(self):
        self.download_data(self.force_download)
        self.unzip_data(self.force_unzip)
        self.process_data(self.force_process)
        print('dataset created')

    def download_data(self, force_download):
        """ Downloads the data from Google Drive """
        
        # Check if the file already exists
        if not os.path.isfile(self.raw_zip_file) or force_download:
            print('Downloading data')
            gdown.download(self.file_url,
                           self.raw_zip_file,
                           quiet=False)
            print('Data successfully downloaded')

    def unzip_data(self, force_unzip):
        """ Unzips the raw data zip file """

        if not os.path.isdir(self.raw_unzipped_file_folder) or force_unzip:
            print('Unzipping data')
            with zipfile.ZipFile(self.raw_zip_file, 'r') as zip_ref:
                zip_ref.extractall(self.raw_unzipped_file_folder)
            print('Data successfully unzipped')

    def process_data(self, force_process):
        """ Processes the raw data """

        if (((not os.path.isfile(self.processed_training_set)) or 
            (not os.path.isfile(self.processed_test_set))) or 
            force_process):
            print('Processing data')
            show_one = 60
            images_array = []
            labels_array = []
            for root, dirs, files in os.walk(self.raw_unzipped_file_folder, topdown=True):
                if(not '__MACOSX' in root):
                    for name in files:
                        if (name.endswith('.png') or 
                            name.endswith('.JPG') or 
                            name.endswith('.JPEG')):

                            # Open the image and change color encoding to RGB 
                            # in case the image is a png
                            image = Image.open(os.path.join(root,name))
                            if (name.endswith('.png')):
                                image = image.convert('RGB')

                            # Perform some transformations on the image 
                            transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((256,256)),
                                    transforms.Normalize((0.5,), (0.5,))])
                            image = transform(image)

                            # Perform kornia operations on the image
                            self.aff = K.RandomAffine(360, return_transform=True, same_on_batch=True)
                            self.cj = K.ColorJitter(0.2, 0.3, 0.2, 0.3)
                            img_out, _ = self.aff(self.cj(image))
                            
                            # Append processed image and label to arrays
                            images_array.append(img_out)
                            labels_array.append(root.split('/')[len(root.split('/'))-1])

            # Create tensors from image array and label array
            if(len(images_array) != 0):
                unique_labels = list(set(labels_array))
                labels_array = [torch.tensor([unique_labels.index(i)]) for i in labels_array]
             
                training_partition = round(len(images_array)*self.training_partition_percentage)
                images_for_training = images_array[0:training_partition]
                labels_for_training = labels_array[0:training_partition]
                images_for_testing = images_array[training_partition:]
                labels_for_testing = labels_array[training_partition:]


                images_for_training_as_tensor = torch.Tensor(len(images_for_training), 3, 256, 256)
                torch.cat(images_for_training, out=images_for_training_as_tensor)
                labels_for_training_as_tensor = torch.Tensor(len(labels_for_training), 1)
                torch.cat(labels_for_training, out=labels_for_training_as_tensor)

                images_for_testing_as_tensor = torch.Tensor(len(images_for_testing), 3, 256, 256)
                torch.cat(images_for_testing, out=images_for_testing_as_tensor)
                labels_for_testing_as_tensor = torch.Tensor(len(labels_for_testing), 1)
                torch.cat(labels_for_testing, out=labels_for_testing_as_tensor)

                torch.save((images_for_training_as_tensor, labels_for_training_as_tensor), self.processed_training_set)
                torch.save((images_for_testing_as_tensor, labels_for_testing_as_tensor), self.processed_test_set)
                torch.save(unique_labels, self.processed_labels)
                print('Finished processing the data')
            else:
                print('No data to process')