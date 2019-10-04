"""
Script parsing dicom metadata and creating dataframe used later to create directory structure.
"""

import os
import pickle
from collections import defaultdict

import pandas
import pydicom
from tqdm import tqdm

from rsna19.config import config

# path under which new directory structure will be created
DF_PATH_OUT = os.path.join(config.data_root, 'df.pkl')


def read_dicom(path):
    return pydicom.dcmread(path, stop_before_pixels=True)


def main():
    tags = ['SOPInstanceUID',
            'Modality',
            'PatientID',
            'StudyInstanceUID',
            'SeriesInstanceUID',
            'StudyID',
            'ImagePositionPatient',
            'ImageOrientationPatient',
            'SamplesPerPixel',
            'PhotometricInterpretation',
            'Rows',
            'Columns',
            'PixelSpacing',
            'BitsAllocated',
            'BitsStored',
            'HighBit',
            'PixelRepresentation',
            'WindowCenter',
            'WindowWidth',
            'RescaleIntercept',
            'RescaleSlope']

    d = defaultdict(list)
    for subset, subset_dir in [('train', config.train_dir), ('test', config.test_dir)]:
        for root, dirs, files in os.walk(subset_dir):
            for file in tqdm(files):
                try:
                    dcm = read_dicom(os.path.join(root, file))
                    for tag in tags:
                        try:
                            d[tag].append(dcm[tag].value)
                        except KeyError:
                            d[tag].append(None)
                    d['path'].append(os.path.join(root, file))
                    d['subset'].append(subset)
                except Exception as e:
                    print(e)
                    print(os.path.join(root, file))

    df = pandas.DataFrame(d)

    with open(DF_PATH_OUT, 'wb') as f:
        pickle.dump(df, f)


if __name__ == '__main__':
    main()
