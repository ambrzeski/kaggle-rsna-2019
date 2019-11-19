import os
import glob
import re
from collections import UserDict

import numpy as np
import nibabel
import pandas as pd

from rsna19.configs.base_config import BaseConfig


MASKS_QUERY = BaseConfig.data_root + "/train/*/Untitled.nii.gz"
SEG_LABEL_MULT = 1


def main():

    labels = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
    new_labels_dict = LabelsFromSegs(MASKS_QUERY)

    csv_root_dir = os.path.normpath(__file__ + '../../../csv')
    data = pd.read_csv(os.path.join(csv_root_dir, '5fold.csv'))

    for exam_id, exam_labels in new_labels_dict.items():
        for i, slice_labels in enumerate(exam_labels):

            path = 'rsna/train/{}/npy/{:03d}.npy'.format(exam_id, i)
            values = data.loc[data.path == path, 'fold'].values
            if len(values) == 0:
                print(f"Could not load entry for: {path}. Skipping.")
                continue

            fold = values[0]

            # Drop last labels for the slice
            # print(data.loc[data['path'] == path].values.tolist()[0])
            old_labels = data.loc[data['path'] == path, labels].values.tolist()[0]
            # print("Old:", old_labels)
            data.drop(data.loc[data['path'] == path].index, inplace=True)

            slice_labels = slice_labels.tolist()
            nonclassified_label = slice_labels.pop()
            # print("New:", slice_labels, nonclassified_label)

            if nonclassified_label:
                print("Non-classified detected. Reverting from {} to {}".format(slice_labels, old_labels))
                slice_labels = old_labels

            # Add new labels
            entry = dict(zip(labels, slice_labels))
            entry.update({'path': path, 'fold': fold})
            for _ in range(SEG_LABEL_MULT):
                data = data.append(entry, ignore_index=True)

                if nonclassified_label:
                    print(entry)
                    print()

    data.to_csv("5fold-rev4.csv", index=False)


class LabelsFromSegs(UserDict):

    CLASSES = [1, 2, 3, 4, 5, 6]
    MASK_SIZE_THRESHOLD = 3

    def __init__(self, query, *args, **kwargs):

        """
        Load new slice labels inferred from segmentation masks

        Is dict like:
        {
            "ID_134d398b61": [
                                [1, 0, 0, 1, 0, 1, 0],     # slice 0 labels
                                [1, 0, 1, 1, 0, 0, 0],     # slice 1 labels
                                ....
                            ],
            ...
        ]

        where labels are in order: [any, epidural, intraparenchymal, intraventricular, subarachnoid, subdural, non-classified]
        """

        super().__init__(*args, **kwargs)
        mask_paths = glob.glob(query)

        for p in mask_paths:
            mask = nibabel.load(p)
            mask_arr = mask.get_fdata()

            labels = []
            slices = np.transpose(mask_arr, (2, 0, 1))

            for slc in slices:
                slice_labels = self._get_labels(slc)
                slice_labels = [any(slice_labels)] + slice_labels
                slice_labels = np.asarray(slice_labels).astype(np.float)
                labels.append(slice_labels)

            res = re.search(r"ID_\w+", p)
            exam_id = res[0]
            self[exam_id] = labels

    def _get_labels(self, slc):
        return [self._is_positive(slc, x) for x in self.CLASSES]

    def _is_positive(self, slc, class_id):
        return np.count_nonzero(slc == class_id) > self.MASK_SIZE_THRESHOLD


if __name__ == '__main__':
    main()
