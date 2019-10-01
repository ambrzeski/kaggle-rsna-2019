import pickle
import pandas as pd

DICOM_TAGS_DF_PATH = '/kolos/m2/ct/data/rsna/df.pkl'
LABELS_PATH = '/kolos/storage/ct/data/rsna/stage_1_train.csv'


def load_dicom_tags():
    with open(DICOM_TAGS_DF_PATH, 'rb') as f:
        df = pickle.load(f)

    return df


def load_labels():
    labels = pd.read_csv(LABELS_PATH)
    labels[['ID', 'Disease']] = labels.ID.str.rsplit("_", 1, expand=True)
    labels = labels[['ID', 'Disease', 'Label']]
    labels = pd.pivot_table(labels, index="ID", columns="Disease", values="Label")

    return labels

