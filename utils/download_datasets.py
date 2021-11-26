import sys
import os
from utils.data_utils import download_from_google_drive
from configs.paths_config import dataset_paths, datasets_path
import kaggle

sys.path.append(".")
sys.path.append("..")

kaggle_datasets = [('xhlulu/flickrfaceshq-dataset-nvidia-resized-256px', dataset_paths['ffhq256']),
                   ('lamsimon/celebahq', dataset_paths['celeba_hq'])]
kaggle.api.authenticate()

for data_name, save_path in kaggle_datasets:
    if not os.listdir(save_path):
        kaggle.api.dataset_download_files(data_name, path=dataset_paths['ffhq256'], unzip=True)



# file_id_name = [('1badu11NqxGf6qM3PTTooQDJvQbejgbTv', 'CelebAMask-HQ.zip')]

print('jajsdas')