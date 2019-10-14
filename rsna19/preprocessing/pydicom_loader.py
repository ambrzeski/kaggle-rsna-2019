import os

import numpy as np
import pydicom

from rsna19.preprocessing.hu_converter import HuConverter


class PydicomLoader:
    def __init__(self):
        cdf = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cdf_vis.npy"))
        self.hu_converter = HuConverter
        self.hu_converter.change_convert_params(cdf, (-400, 2000))

    def window_image(self, img, intercept, slope):
        return img * slope + intercept

    def get_first_of_dicom_field_as_int(self, x):
        # get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
        if type(x) == pydicom.multival.MultiValue:
            return int(x[0])
        else:
            return int(x)

    def get_windowing(self, data):
        dicom_fields = [data[('0028', '1050')].value,  # window center
                        data[('0028', '1051')].value,  # window width
                        data[('0028', '1052')].value,  # intercept
                        data[('0028', '1053')].value]  # slope

        return [self.get_first_of_dicom_field_as_int(x) for x in dicom_fields]

    def load(self, path, convert_hu=True):
        data = pydicom.read_file(path)
        image = data.pixel_array.astype(np.int32)
        window_center, window_width, intercept, slope = self.get_windowing(data)
        image = self.window_image(image, intercept, slope)

        if convert_hu:
            image = self.hu_converter.convert(image)

        return image.astype(np.int16)
