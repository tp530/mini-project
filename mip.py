import os
import numpy as np
from PIL import Image, ImageEnhance
from nptyping import NDArray, Float64, Shape
import random

class Microscopy_image_processor(object):

    @staticmethod
    def load_as_preprocessed_image(full_file_path: str, side_size: int) -> NDArray[Shape["*, *"], Float64]:
        """
        Load a raster image as a normalised center-cropped grayscale image.

        Parameters
        ----------
        full_file_path : str
            A full path to the image, including the file name.

        side_size : int
            The size of a side of the resulting square image (in pixels).

        Returns
        -------
        NDArray[Shape["*, *"], Float64]
            A normalised center-cropped grayscale image.
        """

        if not os.path.isfile(full_file_path):
            raise Exception('Invalid file path.')

        # Load the file as a greyscale image
        image = Image.open(full_file_path).convert("L")

        if image.size[0] < side_size or image.size[1] < side_size:
            raise Exception(
                f"The input image is too small. Minimum image size is {side_size} x {side_size} px.")

        smaller_side = np.array(image.size).min()

        # Center-crop the image to a square
        left = round((image.size[0] - smaller_side) / 2)
        top = round((image.size[1] - smaller_side) / 2)
        right = round((image.size[0] + smaller_side) / 2)
        bottom = top + (right - left)

        image = image.crop((left, top, right, bottom))

        # Resample the image
        image.thumbnail((side_size, side_size), Image.Resampling.BICUBIC)

        # Normalize the image
        img_array = np.asarray(image) / 255.0

        if img_array.shape[0] != side_size or img_array.shape[1] != side_size:
            raise Exception(f"Invalid image shape: {full_file_path}")

        return img_array

    @staticmethod
    def apply_microscopy_visual_atrifacts(image_data: NDArray[Shape["*, *"], Float64], randomness: bool = False) -> NDArray[Shape["*, *"], Float64]:
        """
        Apply brightness and contrast adjustments to the image to create a fluorescence microscopy-like image.

        Parameters
        ----------
        image_data : NDArray[Shape["*, *"]
            A normalised grayscale image.

        randomness : bool
            Determines whether to apply the adjustments of a random intensity. Set the value to False for reproducible results.

        Returns
        -------
        NDArray[Shape["*, *"], Float64]
            A fluorescence microscopy-like image.
        """
        if image_data.ndim != 2:
            raise Exception('Invalid image array size')

        mean = np.mean(image_data)

        if randomness:
            contrast_multiple = 10 + (random.random() * 90)
        else:
            contrast_multiple = 10

        image_data = ((image_data - mean) * contrast_multiple) + mean

        image_data = np.clip(image_data, 0, None) / image_data.max()

        return image_data

    @staticmethod
    def inflate_channels(image_data: NDArray[Shape["*, *"], Float64]) -> NDArray[Shape["*, *, 3"], Float64]:
        """
        Inflates a single-channel image to a grayscale RGB image.

        Parameters
        ----------
        image_data : NDArray[Shape["*, *"]
            A normalised grayscale image.

        Returns
        -------
        NDArray[Shape["*, *"], Float64]
            A grayscale RGB image.
        """

        if image_data.ndim != 2:
            raise Exception("Invalid image array size.")

        image_data = image_data[:, :, np.newaxis]
        inflated = np.repeat(image_data, 3, -1)

        return inflated

    @staticmethod
    def save_image(image_file_path: str, image_data: NDArray[Shape["*, *, 3"], Float64]) -> None:
        """
        Saves a three-channel normalized image array as a JPEG file.

        Parameters
        ----------
        image_data : NDArray[Shape["*, *, 3"]
            A normalised three-channel image.
        """

        if image_data.ndim != 3 or image_data.shape[2] != 3:
            raise Exception("Invalid image array size.")
        im = Image.fromarray((image_data * 255).astype(np.uint8))
        im.save(image_file_path, quality = 100, subsampling = 0)
