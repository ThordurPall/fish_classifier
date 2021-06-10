import gdown
import zipfile
import os.path


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
                 file_url='https://drive.google.com/uc?id=1-xcm54SpglbyFvqEbipEqrmH3GAoNa8R',
                 force_download=False):
        super().__init__()
        self.file_url = file_url
        self.force_download = force_download
        self.raw_zip_folder = './data/raw/'
        self.raw_zip_file = self.raw_zip_folder + 'raw.zip'

    def make_dataset(self):
        self.download_data()
        self.unzip_data()

    def download_data(self):
        """ Downloads the data from Google Drive """

        # Check if the file already exists
        if not os.path.isfile(self.raw_zip_file) or self.force_download:
            gdown.download(self.file_url,
                           self.raw_zip_file,
                           quiet=False)
        print('Data successfully downloaded')
                

    def unzip_data(self):
        """ Unzips the raw data zip file """

        with zipfile.ZipFile(self.raw_zip_file, 'r') as zip_ref:
            zip_ref.extractall(self.raw_zip_folder)
        print('Data successfully unzipped')