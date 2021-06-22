import json
import os
import random
import zipfile
from pathlib import Path

import gdown
import numpy as np
import torch
from PIL import Image

from src.utils.AugmentationPipeline import AugmentationPipeline as ap
from src.utils.DataTransforms import DataTransforms as dt


class MakeDataset:
    """
    A class that handles everything related to getting
    and setting up the training and test datasets

    ...

    Methods
    -------
    download_data()
        Downloads the data from Google Drive and unzips it
    """

    def __init__(
        self,
        file_url="https://drive.google.com/uc?id=1WyVALHmHhPdt5e-XLSgFZ8aeg6dSI0dI",
        force_download=False,
        force_unzip=False,
        force_process=False,
        training_partition_percentage=0.85,
        generated_images_per_image=1,
        image_size=128,
        add_csv_file=False,
    ):
        super().__init__()
        project_dir = Path(__file__).resolve().parents[2]
        self.file_url = file_url
        self.force_download = force_download
        self.force_unzip = force_unzip
        self.force_process = force_process
        self.data_folder = str(project_dir) + "/data"
        self.raw_unzipped_file_folder = str(project_dir) + "/data/raw/unzipped"
        self.raw_zip_file = str(project_dir) + "/data/raw/raw.zip"
        self.processed_training_set = str(project_dir) + "/data/processed/training.pt"
        self.processed_test_set = str(project_dir) + "/data/processed/test.pt"
        self.processed_training_set_csv = (
            str(project_dir) + "/data/processed/training.csv"
        )
        self.processed_test_set_csv = str(project_dir) + "/data/processed/test.csv"
        self.processed_files_folder = str(project_dir) + "/data/processed"
        self.mapping_file = str(project_dir) + "/data/processed/mapping.json"
        self.raw_files_folder = str(project_dir) + "/data/raw"
        self.training_partition_percentage = training_partition_percentage
        self.generated_images_per_image = generated_images_per_image
        self.image_size_x = image_size
        self.image_size_y = image_size
        self.add_csv_file = add_csv_file

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
        self.process_data(self.force_process, self.add_csv_file)
        print("dataset created")

    def download_data(self, force_download):
        """Downloads the data from Google Drive"""

        # Check if the file already exists
        if not os.path.isfile(self.raw_zip_file) or force_download:
            print("Downloading data")
            # Download the file
            gdown.download(self.file_url, self.raw_zip_file, quiet=False)
            print("Data successfully downloaded")

    def unzip_data(self, force_unzip):
        """Unzips the raw data zip file"""
        # Check if the folder already exists
        if not os.path.isdir(self.raw_unzipped_file_folder) or force_unzip:
            print("Unzipping data")
            # Unzip the data
            with zipfile.ZipFile(self.raw_zip_file, "r") as zip_ref:
                zip_ref.extractall(self.raw_unzipped_file_folder)
            print("Data successfully unzipped")

    def process_data(self, force_process, add_csv_file):
        """Processes the raw data"""

        if (
            (not os.path.isfile(self.processed_training_set))
            or (not os.path.isfile(self.processed_test_set))
        ) or force_process:
            print("Processing data")
            images_array = []
            labels_array = []
            for root, dirs, files in os.walk(
                self.raw_unzipped_file_folder, topdown=True
            ):
                if "__MACOSX" not in root:
                    for name in files:
                        if (
                            name.endswith(".png")
                            or name.endswith(".JPG")
                            or name.endswith(".JPEG")
                        ):

                            # Open the image and change color encoding to RGB
                            # in case the image is a png
                            image = Image.open(os.path.join(root, name))

                            # Perform some transformations on the image
                            image = dt().PIL_image_to_tensor(image).unsqueeze(0)

                            # Perform augumentations on the image
                            for _ in range(self.generated_images_per_image):
                                # Perform kornia operations on the image
                                img_out = ap().forward(image)

                                # Append processed image and label to arrays
                                images_array.append(img_out)
                                labels_array.append(
                                    root.split("/")[len(root.split("/")) - 1]
                                )

            # Process the data to correct data types and save the data
            if len(images_array) != 0:
                # Get all unique labels
                unique_labels = list(set(labels_array))

                # Convert strings in labels array to integers
                labels_array = [
                    torch.tensor([unique_labels.index(label)]) for label in labels_array
                ]

                # Create the mapping (integer: string)
                mapping_dict = dict(
                    zip(
                        [int(unique_labels.index(label)) for label in unique_labels],
                        unique_labels,
                    )
                )

                # Shuffle the data
                mapIndexPosition = list(zip(images_array, labels_array))
                random.shuffle(mapIndexPosition)
                images_array, labels_array = zip(*mapIndexPosition)

                # Convert the data into a training set and a test set
                training_partition = round(
                    len(images_array) * self.training_partition_percentage
                )
                images_for_training = images_array[0:training_partition]
                labels_for_training = labels_array[0:training_partition]
                images_for_testing = images_array[training_partition:]
                labels_for_testing = labels_array[training_partition:]

                if add_csv_file:
                    self.add_csv_file(
                        images_for_training,
                        labels_for_training,
                        images_for_testing,
                        labels_for_testing,
                    )

                # Convert images and labels in training set to tensors
                images_for_training_as_tensor = torch.Tensor(
                    len(images_for_training), 3, self.image_size_x, self.image_size_y
                )
                torch.cat(images_for_training, out=images_for_training_as_tensor)
                labels_for_training_as_tensor = torch.Tensor(
                    len(labels_for_training), 1
                )
                torch.cat(labels_for_training, out=labels_for_training_as_tensor)

                # Convert images and labels in test set to tensors
                images_for_testing_as_tensor = torch.Tensor(
                    len(images_for_testing), 3, self.image_size_x, self.image_size_y
                )
                torch.cat(images_for_testing, out=images_for_testing_as_tensor)
                labels_for_testing_as_tensor = torch.Tensor(len(labels_for_testing), 1)
                torch.cat(labels_for_testing, out=labels_for_testing_as_tensor)

                # Save training and test set
                torch.save(
                    (images_for_training_as_tensor, labels_for_training_as_tensor),
                    self.processed_training_set,
                )
                torch.save(
                    (images_for_testing_as_tensor, labels_for_testing_as_tensor),
                    self.processed_test_set,
                )

                # Save the mapping dict
                with open(self.mapping_file, "w", encoding="utf-8") as f:
                    json.dump(mapping_dict, f, ensure_ascii=False, indent=4)

                print("Finished processing the data")
            else:
                print("No data to process")

    def add_csv_file(
        self,
        images_for_training,
        labels_for_training,
        images_for_testing,
        labels_for_testing,
    ):
        pixels = (
            images_for_training[0].shape[1]
            * images_for_training[0].shape[2]
            * images_for_training[0].shape[3]
        )
        csv_data_training_set = np.array(
            np.append(
                [i + 0.0 for i in range(pixels)],
                [pixels + 0.0],
            )
        )
        csv_data_testing_set = np.array(
            np.append(
                [i + 0.0 for i in range(pixels)],
                [pixels + 0.0],
            )
        )
        for i, image in enumerate(images_for_training):
            csv_data_training_set = np.vstack(
                (
                    csv_data_training_set,
                    [
                        np.append(
                            np.array(image).ravel(),
                            np.array(labels_for_training[i]),
                        )
                    ],
                )
            )
        for i, image in enumerate(images_for_testing):
            csv_data_testing_set = np.vstack(
                (
                    csv_data_testing_set,
                    [
                        np.append(
                            np.array(image).ravel(),
                            np.array(labels_for_testing[i]),
                        )
                    ],
                )
            )
        for idx, x in np.ndenumerate(csv_data_training_set):
            if idx[0] != 0:
                csv_data_training_set[idx[0]][idx[1]] = int(x * 255.0)
        for idx, x in np.ndenumerate(csv_data_testing_set):
            if idx[0] != 0:
                csv_data_testing_set[idx[0]][idx[1]] = int(x * 255.0)
        np.savetxt(
            self.processed_training_set_csv,
            csv_data_training_set,
            delimiter=",",
            fmt="%d",
        )
        np.savetxt(
            self.processed_test_set_csv,
            csv_data_testing_set,
            delimiter=",",
            fmt="%d",
        )
