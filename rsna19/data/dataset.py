import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from rsna19.data.utils import normalize_train


class IntracranialDataset(Dataset):
    # _NUM_SLICES must be odd
    _NUM_SLICES = 3  # TODO get this from config maybe
    _SLICE_SIZE = 512

    def __init__(self, config, folds, return_labels=True):
        """
        :param csv_file: path to csv file
        :param folds: list of selected folds
        :param csv_root_dir: prepended to csv_file path, defaults to project's rsna19/datasets
        :param return_labels: if True, labels will be returned with image
        :param preprocess_func: preprocessing function, e.g. for window adjustment
        """
        self.config = config
        self.return_labels = return_labels

        if config.csv_root_dir is None:
            csv_root_dir = os.path.normpath(__file__ + '/../../datasets')
        else:
            csv_root_dir = config.csv_root_dir

        data = pd.read_csv(os.path.join(csv_root_dir, config.dataset_file))
        data = data[data.fold.isin(folds)]
        data = data.reset_index()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        middle_img_path = Path(os.path.normpath(os.path.join(self.config.data_root, '..', self.data.loc[idx, 'path'])))
        middle_img_num = int(middle_img_path.stem)
        slices_image = np.zeros((self._NUM_SLICES, self._SLICE_SIZE, self._SLICE_SIZE))
        for idx, img_num in enumerate(range(middle_img_num - self._NUM_SLICES//2,
                                            middle_img_num + self._NUM_SLICES//2 + 1)):
            if img_num < 0:
                img_num = 0
            if img_num > (len(os.listdir(middle_img_path.parent)) - 1):
                img_num = len(os.listdir(middle_img_path.parent)) - 1

            # [:512, :512] temporary workaround for bigger images
            slices_image[idx] = np.load(middle_img_path.parent.joinpath('{:03d}.npy'.format(img_num)))[:512, :512]

        if normalize_train:
            slices_image = normalize_train(slices_image,
                                           self.config.min_hu_value,
                                           self.config.max_hu_value)

        img = torch.tensor(slices_image, dtype=torch.float32)

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
