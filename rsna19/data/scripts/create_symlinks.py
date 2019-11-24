"""
Script creating directory structure based on dataframe with dicom metadata.
New dircetory structure contains symlinks to original dicom files, grouped by StudyInstanceUID:
<root>/<train/test>/<StudyInstanceUID>/dicom/<slice_ix>.dcm
"""

import os
import pickle
from collections import defaultdict

import numpy as np
import pandas
from tqdm import tqdm

from rsna19.configs.base_config import BaseConfig

DF_PATHS_IN = [
    os.path.join(BaseConfig.data_root, 'df.pkl'),
    os.path.join(BaseConfig.data_root, 'df-stage2-test.pkl')]

# dataframe mapping SOPInstanceUID of each dicom to path in new directory structure
ID_DF_PATHS_OUT = [
    os.path.join(BaseConfig.data_root, 'id_to_path.pkl'),
    os.path.join(BaseConfig.data_root, 'id_to_path_stage2-test.pkl')]


def main(df_path_in, id_df_path_out):
    with open(df_path_in, 'rb') as f:
        df = pickle.load(f)

    id_to_path = defaultdict(list)

    for study_id, dicoms_df in tqdm(
            df[['SOPInstanceUID', 'StudyInstanceUID', 'ImagePositionPatient', 'subset', 'path']].groupby(
                'StudyInstanceUID')):

        subset = dicoms_df.iloc[0].subset
        study_dir = os.path.join(BaseConfig.data_root, subset, study_id, 'dicom')
        os.makedirs(study_dir)

        # sort by z value
        dicoms_df = dicoms_df.iloc[np.argsort([float(pos[2]) for pos in dicoms_df.ImagePositionPatient])]

        for i, row in enumerate(dicoms_df.itertuples()):
            target_path = row.path
            link_name = os.path.join(study_dir, f'{i:03d}.dcm')
            os.symlink(target_path, link_name)

            id_to_path['SOPInstanceUID'].append(row.SOPInstanceUID)
            id_to_path['path'].append(link_name)

    id_to_path = pandas.DataFrame(id_to_path)
    id_to_path.to_pickle(id_df_path_out)


if __name__ == '__main__':
    for path_in, path_out in zip(DF_PATHS_IN, ID_DF_PATHS_OUT):
        main(path_in, path_out)
