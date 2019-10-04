import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class IntracranialDataset(Dataset):

    def __init__(self, csv_file, basepath, return_labels, folds, preprocess_func=None):
        """
        :param csv_file: path to csv file
        :param basepath: root dataset path, prepended to paths from csv file
        :param return_labels: if True, labels will be returned with image
        :param folds: list of selected folds
        :param preprocess_func: preprocessing function, e.g. for window adjustment
        """

        self.basepath = basepath
        self.return_labels = return_labels
        self.preprocess_func = preprocess_func

        data = pd.read_csv(csv_file)
        data = data[data.fold.isin(folds)]
        data = data.reset_index()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_path = os.path.join(self.basepath, self.data.loc[idx, 'path'])
        img = np.load(img_path)

        if self.preprocess_func:
            img = self.preprocess_func(img)

        if self.return_labels:
            labels = torch.tensor(self.data.loc[idx, ['epidural',
                                                      'intraparenchymal',
                                                      'intraventricular',
                                                      'subarachnoid',
                                                      'subdural',
                                                      'any']])
            return {'image': img, 'labels': labels}

        else:
            return {'image': img}
