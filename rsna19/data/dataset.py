import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from rsna19.config import config


class IntracranialDataset(Dataset):

    def __init__(self, csv_file, folds, csv_root_dir=None, return_labels=True, preprocess_func=None):
        """
        :param csv_file: path to csv file
        :param folds: list of selected folds
        :param csv_root_dir: prepended to csv_file path, defaults to project's rsna19/datasets
        :param return_labels: if True, labels will be returned with image
        :param preprocess_func: preprocessing function, e.g. for window adjustment
        """

        self.return_labels = return_labels
        self.preprocess_func = preprocess_func

        if csv_root_dir is None:
            csv_root_dir = os.path.normpath(__file__ + '/../../datasets')

        data = pd.read_csv(os.path.join(csv_root_dir, csv_file))
        data = data[data.fold.isin(folds)]
        data = data.reset_index()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_path = os.path.normpath(os.path.join(config.data_root, '..', self.data.loc[idx, 'path']))
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
