import os
import pickle
import pandas as pd

from rsna19.configs.base_config import BaseConfig

DICOM_TAGS_DF_PATH = '/kolos/m2/ct/data/rsna/df.pkl'


def load_dicom_tags():
    with open(DICOM_TAGS_DF_PATH, 'rb') as f:
        df = pickle.load(f)

    return df


def load_labels():
    labels = pd.read_csv(BaseConfig.labels_path)
    labels[['SOPInstanceUID', 'Disease']] = labels.ID.str.rsplit("_", 1, expand=True)
    labels = labels[['SOPInstanceUID', 'Disease', 'Label']]
    labels = pd.pivot_table(labels, index="SOPInstanceUID", columns="Disease", values="Label")

    return labels


def load_df_with_labels_and_dicom_tags():
    tags = load_dicom_tags()
    labels = load_labels()

    return labels.merge(tags, on='SOPInstanceUID', how='outer')


def normalize_train(image, min_hu_value=-1000, max_hu_value=1000):
    """normalize hu values to -1 to 1 range"""
    image[image < min_hu_value] = min_hu_value
    image[image > max_hu_value] = max_hu_value
    image = (image - min_hu_value) / ((max_hu_value - min_hu_value) / 2) - 1
    return image
