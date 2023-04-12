import os
import urllib.request
import tarfile
import shutil
import random
import numpy as np
from nptyping import NDArray, Float64, Shape

from generator import Zernike_generator
from mip import Microscopy_image_processor

class Dataset(object):

    @staticmethod
    def download_if_missing(local_data_folder: str, dataset_ulr: str) -> None:

        local_data_folder = os.path.join(os.path.abspath(os.getcwd()), "data")
        dataset_ulr = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"

        if not os.path.exists(local_data_folder):
            os.mkdir(local_data_folder)

        if not os.path.isdir(local_data_folder):
            raise Exception(f"{local_data_folder} is not a folder.")

        # Delete the .DS_Store file
        ds_store = os.path.join(local_data_folder, ".DS_Store")
        if os.path.exists(ds_store) and os.path.isfile(ds_store):
            os.remove(ds_store)

        # Delete .directory
        dot_dir = os.path.join(local_data_folder, ".directory")
        if os.path.exists(dot_dir) and os.path.isfile(dot_dir):
            os.remove(dot_dir)

        files = [x[2] for x in os.walk(local_data_folder)][0]
        if len(files) != 5640:
            # Delete the old files
            for file in files:
                image_file_path = os.path.join(local_data_folder, file)
                os.remove(image_file_path)

            # Download the dataset
            print("Downloading the dataset. Please wait.")
            tar_file_path = os.path.join(local_data_folder, "dtd-r1.0.1.tar.gz")
            urllib.request.urlretrieve(dataset_ulr, tar_file_path)
            tar_file = tarfile.open(tar_file_path)
            tar_file.extractall(local_data_folder)
            tar_file.close()

            # Move the files
            extracted_images_folder = os.path.join(os.path.join(local_data_folder, "dtd"), "images")
            dtd_folders = [x[0] for x in os.walk(extracted_images_folder)]
            for dtd_folder in dtd_folders:
                dtd_files = [x[2] for x in os.walk(dtd_folder)][0]
                for dtd_file in dtd_files:
                    dtd_file_path = os.path.join(os.path.join(extracted_images_folder, dtd_folder), dtd_file)
                    new_file_path = os.path.join(local_data_folder, dtd_file)
                    os.replace(dtd_file_path, new_file_path)
            shutil.rmtree(os.path.join(local_data_folder, "dtd"), ignore_errors = True)

            os.remove(tar_file_path)

            if os.path.exists(dot_dir) and os.path.isfile(dot_dir):
                os.remove(dot_dir)

            new_files = [x[2] for x in os.walk(local_data_folder)][0]
            if len(new_files) == 5640:
                print("The dataset is ready. All 5,640 files are available.")
            else:
                print("Something went wrong when downloading the dataset.")

        else:
            print("The dataset is present. Skipping download.")

    @staticmethod
    def get_features_and_labels(data_folder: str, sample_side_size: int):

        # Delete the .DS_Store file
        ds_store = os.path.join(data_folder, ".DS_Store")
        if os.path.exists(ds_store) and os.path.isfile(ds_store):
            os.remove(ds_store)

        # Delete .directory
        dot_dir = os.path.join(data_folder, ".directory")
        if os.path.exists(dot_dir) and os.path.isfile(dot_dir):
            os.remove(dot_dir)

        files = [x[2] for x in os.walk(f"{data_folder}")][0]
        number_of_samples = len(files)
        features = np.empty((number_of_samples, sample_side_size, sample_side_size, 3), dtype=float)
        labels = np.empty(number_of_samples, dtype=float)
        for index, file in enumerate(files):
            image_file_path = os.path.join(data_folder, file)
            defocus_weight = random.random()
            generator = Zernike_generator(64, 1)
            generator.add_aberration(2, 0, defocus_weight)
            image = Microscopy_image_processor.load_as_preprocessed_image(image_file_path, sample_side_size)
            image = Microscopy_image_processor.apply_microscopy_visual_atrifacts(image)
            image = generator.apply_psf_to_image(image)
            image = Microscopy_image_processor.inflate_channels(image)
            features[index] = image
            labels[index] = defocus_weight
        return (features, labels)

    @staticmethod
    def get_single_sample(image_file_path: str, defocus_weight: float, sample_side_size: int) -> NDArray[Shape["1, *, *, 3"], Float64]:
        features = np.empty((1, sample_side_size, sample_side_size, 3), dtype=float)
        generator = Zernike_generator(64, 1)
        generator.add_aberration(2, 0, defocus_weight)
        image = Microscopy_image_processor.load_as_preprocessed_image(image_file_path, sample_side_size)
        image = Microscopy_image_processor.apply_microscopy_visual_atrifacts(image)
        image = generator.apply_psf_to_image(image)
        image = Microscopy_image_processor.inflate_channels(image)
        features[0] = image
        return features
