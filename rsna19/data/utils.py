import pickle
import pandas as pd

# DICOM_TAGS_DF_PATH = '/kolos/m2/ct/data/rsna/df.pkl'
# LABELS_PATH = '/kolos/storage/ct/data/rsna/stage_1_train.csv'
DICOM_TAGS_DF_PATH = '/home/tomek/data/df.pkl'
LABELS_PATH = '/home/tomek/data/stage_1_train.csv'


def load_dicom_tags():
    with open(DICOM_TAGS_DF_PATH, 'rb') as f:
        df = pickle.load(f)

    return df


def load_labels():
    labels = pd.read_csv(LABELS_PATH)
    labels[['SOPInstanceUID', 'Disease']] = labels.ID.str.rsplit("_", 1, expand=True)
    labels = labels[['SOPInstanceUID', 'Disease', 'Label']]
    labels = pd.pivot_table(labels, index="SOPInstanceUID", columns="Disease", values="Label")

    return labels


def load_df_with_labels_and_dicom_tags():
    tags = load_dicom_tags()
    labels = load_labels()

    return labels.merge(tags, on='SOPInstanceUID', how='outer')
