import os
import numpy as np
from scipy.interpolate import interp1d


class HuConverter:
    """Class for converting HU units to 8 bits or number of bits specified in cdf"""
    min_hu_value = -400
    max_hu_value = 1000

    cdf = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cdf.npy"))
    window = (min_hu_value, max_hu_value)

    @classmethod
    def change_convert_params(cls, cdf, window):
        cls.cdf = cdf
        cls.window = window

    @classmethod
    def convert(cls, image, use_cdf=True):
        hu_min, hu_max = cls.window

        if not use_cdf:
            image = cls._hu_to_8bits(image, min_hu_value=hu_min, max_hu_value=hu_max)
        else:
            image = cls._hu_convert_with_cdf(image, cls.cdf, min_hu_value=hu_min, max_hu_value=hu_max)

        return image

    @staticmethod
    def _hu_convert_with_cdf(slice_pixels, cdf, min_hu_value=-400, max_hu_value=400):
        """Convert pixel values (MIN_HU_VALUE, MAX_HU_VALUE) using cdf, number of bits specified in cdf

        :param slice_pixels: nD numpy array with a single slice
        :return: converted nD numpy array
        """
        slice_pixels_copy = slice_pixels.copy().astype(np.int32)
        slice_pixels_copy[slice_pixels_copy < min_hu_value] = min_hu_value
        slice_pixels_copy[slice_pixels_copy > max_hu_value] = max_hu_value
        slice_pixels_copy -= min_hu_value
        return cdf[slice_pixels_copy]

    @staticmethod
    def _hu_to_8bits(slice_pixels, min_hu_value=-400, max_hu_value=400):
        """Convert pixel values (MIN_HU_VALUE, MAX_HU_VALUE) to 8 bits (available to save in PNG)

        :param slice_pixels: 2D numpy array with a single slice
        :return: 2D numpy array converted to 8 bits
        """
        slice_pixels_copy = slice_pixels.copy()
        slice_pixels_copy[slice_pixels_copy < min_hu_value] = min_hu_value
        slice_pixels_copy[slice_pixels_copy > max_hu_value] = max_hu_value
        converter = interp1d([min_hu_value, max_hu_value], [0, 255])
        return np.uint8(converter(slice_pixels_copy))

