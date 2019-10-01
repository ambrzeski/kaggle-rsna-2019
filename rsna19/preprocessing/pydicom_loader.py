import pydicom

from rsna19.preprocessing.hu_converter import HuConverter


class PydicomLoader:
    def __init__(self):
        self.hu_converter = HuConverter()

    def window_image(self, img, window_center, window_width, intercept, slope):
        img = (img * slope + intercept)
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img[img < img_min] = img_min
        img[img > img_max] = img_max
        return img

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

    def load(self, path):
        data = pydicom.read_file(path)
        image = data.pixel_array
        window_center, window_width, intercept, slope = self.get_windowing(data)
        image = self.window_image(image, window_center, window_width, intercept, slope)
        image = self.hu_converter.convert(image)

        return image
