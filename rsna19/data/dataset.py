import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from rsna19.configs.base_config import BaseConfig


class IntracranialDataset(Dataset):
    # _NUM_SLICES must be odd
    _NUM_SLICES = 3  # TODO get this from config maybe
    _SLICE_SIZE = 512

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

        middle_img_path = Path(os.path.normpath(os.path.join(BaseConfig.data_root, '..', self.data.loc[idx, 'path'])))
        middle_img_num = int(middle_img_path.stem)
        slices_image = np.zeros((self._NUM_SLICES, self._SLICE_SIZE, self._SLICE_SIZE))
        for idx, img_num in enumerate(range(middle_img_num - self._NUM_SLICES//2, middle_img_num - self._NUM_SLICES//2)):
            # [:512, :512] temporary workaround for bigger images
            print(middle_img_path.parent.joinpath('{}.npy'.format(img_num)))
            slices_image[idx] = np.load(middle_img_path.parent.joinpath('{}.npy'.format(img_num)))[:512, :512]

        img = torch.tensor(slices_image, dtype=torch.float32)

        if self.preprocess_func:
            img = self.preprocess_func(img)

        if self.return_labels:
            labels = torch.tensor(self.data.loc[idx, ['epidural',
                                                      'intraparenchymal',
                                                      'intraventricular',
                                                      'subarachnoid',
                                                      'subdural',
                                                      'any']], dtype=torch.float32)
            return {'image': img, 'labels': labels}

        else:
            return {'image': img}
