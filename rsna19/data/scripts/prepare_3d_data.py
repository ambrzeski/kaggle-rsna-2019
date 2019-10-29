""" Load dicom files using vtk package """
import json

import shutil

import os
from glob import glob
from math import atan
from collections import namedtuple
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path

import numpy as np
import vtk
from scipy import ndimage
from vtk import vtkImageCast, vtkImageResample, vtkDICOMImageReader
from vtk.util.numpy_support import vtk_to_numpy
import tqdm

from rsna19.configs.base_config import BaseConfig
from rsna19.data.utils import crop_scan
from rsna19.preprocessing.pydicom_loader import PydicomLoader

ShearParams = namedtuple('ShearParams', 'rad_tilt, minus_center_z')
OUT_SIZE = (400, 400)
BG_HU = -2000

loader = PydicomLoader()


class VtkImage:
    """Class for loading and scaling dicoms using Vtk library"""

    def __init__(self, scan_dir, spacing='auto'):
        """Load image to scale
        :param scan_dir: path to dicom file
        :param spacing: [x,y,z] spacing in mm, or 'auto' if we want to use min spacing already present in a scan,
               'none' if we are not doing any resamplig
        """

        # read dicom
        self.reader = vtkDICOMImageReader()
        self.reader.ReleaseDataFlagOff()
        self.reader.SetDirectoryName(scan_dir)
        self.reader.Update()

        # prepare parameters for shear transform (gantry tilt)
        x1, y1, z1, x2, y2, z2 = self.image_orientation = self.reader.GetImageOrientationPatient()

        # if non-standard orientation, then it's non-standard series
        if y2 == 0:
            raise Exception(f"Wrong patient orientation: {self.image_orientation}")

        rad_tilt = atan(z2 / y2)
        center_z = self.reader.GetOutput().GetBounds()[5] / 2
        self.shear_params = ShearParams(rad_tilt, -center_z)

        self.scan_dir = scan_dir
        self.spacing = spacing

        self.angle_z = 0
        self.angle_y = 0
        self.origin_x = None
        self.image = None

    def set_transform(self, angle_z, angle_y, origin_x):
        self.angle_z = angle_z
        self.angle_y = angle_y
        self.origin_x = origin_x
        self.image = None

    def set_spacing(self, spacing):
        self.spacing = spacing
        self.image = None

    def update_image(self):
        reslice = vtk.vtkImageReslice()

        if self.origin_x is not None:
            # add padding so that origin_x is in the middle of the image
            pad = vtk.vtkImageConstantPad()
            pad.SetInputConnection(self.reader.GetOutputPort())
            pad.SetConstant(BG_HU)

            # GetExtent() returns a tuple (minX, maxX, minY, maxY, minZ, maxZ)
            extent = list(self.reader.GetOutput().GetExtent())
            x_size = extent[1] - extent[0]
            extent[0] -= max(x_size - 2 * self.origin_x, 0)
            extent[1] += max(2 * self.origin_x - x_size, 0)
            pad.SetOutputWholeExtent(*extent)
            reslice.SetInputConnection(pad.GetOutputPort())
        else:
            reslice.SetInputConnection(self.reader.GetOutputPort())

        transform = vtk.vtkPerspectiveTransform()

        # gantry tilt
        transform.Shear(0, *self.shear_params)

        if self.angle_z != 0 or self.angle_y != 0:
            transform.RotateWXYZ(-self.angle_z, 0, 0, 1)  # top
            transform.RotateWXYZ(self.angle_y, 0, 1, 0)  # front

        reslice.SetResliceTransform(transform)
        reslice.SetInterpolationModeToCubic()
        reslice.AutoCropOutputOn()
        reslice.SetBackgroundLevel(BG_HU)
        reslice.Update()

        spacings_lists = reslice.GetOutput().GetSpacing()

        if self.spacing == 'auto':
            min_spacing = min(spacings_lists)
            if not min_spacing:
                raise ValueError('Invalid scan. Path: {}'.format(self.scan_dir))
            spacing = [min_spacing, min_spacing, min_spacing]

        elif self.spacing == 'none':
            spacing = None
        else:
            spacing = self.spacing

        if spacing is None:
            self.image = reslice
        else:
            resample = vtkImageResample()
            resample.SetInputConnection(reslice.GetOutputPort())
            resample.SetAxisOutputSpacing(0, spacing[0])  # x axis
            resample.SetAxisOutputSpacing(1, spacing[1])  # y axis
            resample.SetAxisOutputSpacing(2, spacing[2])  # z axis
            resample.SetInterpolationModeToCubic()
            resample.Update()

            self.image = resample

    def get_slices(self, dtype=np.float32):
        """Function that returns all slices in original size after gantry tilt handling"""

        if self.image is None:
            self.update_image()

        image = self.image.GetOutput()
        rows, cols, depth = image.GetDimensions()
        spacing = image.GetSpacing()

        scalars = image.GetPointData().GetScalars()
        array = vtk_to_numpy(scalars)

        array = array.reshape(depth, cols, rows)
        array = np.rot90(array, 2, axes=(0, 1))

        if dtype:
            array = array.astype(dtype)

        if len(array) < 5:
            raise Exception("Cannot read 3D dicom image")

        return array, spacing, self.shear_params


def process_scan(scan_dir):
    out_dir = scan_dir.replace('dicom/', '3d/')
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    try:
        vtk_image = VtkImage(scan_dir, spacing='none')
        scan, spacing, _ = vtk_image.get_slices()
        image_orientation = vtk_image.image_orientation

    except Exception:
        traceback.print_exc()
        print(scan_dir)

        exam_root = Path(scan_dir)
        slices = []
        for slice_path in sorted(exam_root.iterdir()):
            slices.append(loader.load(str(slice_path), convert_hu=False))
        scan = np.stack(slices)
        spacing = None
        image_orientation = None

    _, y, x = ndimage.measurements.center_of_mass(scan > 0)
    pre_crop_shape = scan.shape
    scan_cropped = crop_scan(scan, OUT_SIZE, x, y, BG_HU)

    meta = {
        'spacing': spacing,
        'image_orientation': image_orientation,
        'crop_x': x,
        'crop_y': y,
        'pre_crop_shape': pre_crop_shape,
        'out_shape': scan_cropped.shape
    }
    with open(out_dir + '../meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    for idx, scan_slice in enumerate(scan_cropped):
        np.save(f'{out_dir}{idx:03d}.npy', scan_slice.astype(np.int16))


def main():
    with ProcessPoolExecutor(max_workers=16) as executor:
        paths = glob(f'{BaseConfig.data_root}/train/*/dicom/') + glob(f'{BaseConfig.data_root}/test/*/dicom/')
        # paths = ['/kolos/m2/ct/data/rsna/train/ID_0dc7645c14/dicom/',
        #          '/kolos/m2/ct/data/rsna/train/ID_fcc85d6ebc/dicom/',
        #          '/kolos/m2/ct/data/rsna/test/ID_be88c23b42/dicom/',
        #          '/kolos/m2/ct/data/rsna/train/ID_99e4e009da/dicom/',
        #          '/kolos/m2/ct/data/rsna/train/ID_f03b07e5e4/dicom/',
        #          '/kolos/m2/ct/data/rsna/train/ID_c9a49565ec/dicom/',
        #          '/kolos/m2/ct/data/rsna/train/ID_3bb2b2176b/dicom/',
        #          '/kolos/m2/ct/data/rsna/test/ID_a06f6350dc/dicom/']
        list(tqdm.tqdm(executor.map(process_scan, paths), total=len(paths)))


if __name__ == '__main__':
    main()
