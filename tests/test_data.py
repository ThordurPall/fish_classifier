# -*- coding: utf-8 -*-
import os
import glob
import torch
from src.data.MakeDataset import MakeDataset


class TestData:
    def test_download_data(self):
        """
        Test that downloading the dataset from Google Drive works as expected
        """
        make_data = MakeDataset()
        make_data.download_data(True)

        # Test that the data has been downloaded
        assert os.path.isfile(make_data.raw_zip_file)

    def test_unzip_data(self):
        """
        Test that unzipping the dataset works as expected
        """
        make_data = MakeDataset()
        make_data.download_data(False)
        make_data.unzip_data(True)

        # Test that the unzipped folder exists
        assert os.path.isdir(make_data.raw_unzipped_file_folder)

        # Check that all the 430 pictures have been unzipped
        zip_file = make_data.raw_unzipped_file_folder + '/NA_Fish_Dataset'
        assert len(glob.glob(zip_file + '/*/*.png')) == 281
        assert len(glob.glob(zip_file + '/*/*.JPG')) == 148
        assert len(glob.glob(zip_file + '/*/*.jpeg')) == 1
        
