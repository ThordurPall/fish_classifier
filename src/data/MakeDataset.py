import gdown
import zipfile
from pathlib import Path

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

    def __init__(self, file_url):
        super().__init__()
        self.file_url = file_url


    def download_unzip_data(self):
        """ Downloads the data from Google Drive and unzips it """

        project_dir = Path(__file__).resolve().parents[2]
        print(project_dir)
        gdown.download(self.file_url,
                       './data/raw/raw.zip',
                       quiet=False)
        #with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        #    zip_ref.extractall(directory_to_extract_to)
        #!unzip ./horse2zebra.zip > /dev/null