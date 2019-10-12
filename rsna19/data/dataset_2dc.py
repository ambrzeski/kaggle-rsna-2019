import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations
import cv2

from rsna19.data.utils import normalize_train
from rsna19.preprocessing.hu_converter import HuConverter


class IntracranialDataset(Dataset):
    _HU_AIR = -1000

    def __init__(self, config, folds, return_labels=True, augment=False):
        """
        :param csv_file: path to csv file
        :param folds: list of selected folds
        :param csv_root_dir: prepended to csv_file path, defaults to project's rsna19/datasets
        :param return_labels: if True, labels will be returned with image
        :param preprocess_func: preprocessing function, e.g. for window adjustment
        """
        self.config = config
        self.return_labels = return_labels
        self.augment = augment

        if config.csv_root_dir is None:
            csv_root_dir = os.path.normpath(__file__ + '/../csv')
        else:
            csv_root_dir = config.csv_root_dir

        data = pd.read_csv(os.path.join(csv_root_dir, config.dataset_file))
        data = data[data.fold.isin(folds)]
        data = data.reset_index()
        self.data = data

        if self.config.use_cdf:
            self.hu_converter = HuConverter

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = os.path.normpath(os.path.join(self.config.data_root, '..', self.data.loc[idx, 'path']))

        # todo it would be better to have generic paths in csv and parameter specifying which data version to use
        if self.config.slice_size == 256:
            path = path.replace('npy/', 'npy256/')

        middle_img_path = Path(path)

        middle_img_num = int(middle_img_path.stem)
        slices_image = np.zeros((self.config.num_slices, self.config.slice_size, self.config.slice_size))
        for slice_idx, img_num in enumerate(range(middle_img_num - self.config.num_slices // 2,
                                                  middle_img_num + self.config.num_slices // 2 + 1)):

            if img_num < 0 or img_num > (len(os.listdir(middle_img_path.parent)) - 1):
                slice_img = np.full((self.config.slice_size, self.config.slice_size), self._HU_AIR)
            else:
                slice_img = np.load(middle_img_path.parent.joinpath('{:03d}.npy'.format(img_num)))

            # temporary workaround for bigger images
            if slice_img.shape != (self.config.slice_size, self.config.slice_size):
                slice_img = slice_img[:self.config.slice_size, :self.config.slice_size]
            slices_image[slice_idx] = slice_img

        if self.config.use_cdf:
            slices_image = self.hu_converter.convert(slices_image)
            slices_image = slices_image / (np.iinfo(np.uint16).max / 2) - 1
        else:
            slices_image = normalize_train(slices_image,
                                           self.config.min_hu_value,
                                           self.config.max_hu_value)

        if self.augment:
            augmentations = albumentations.Compose([
                albumentations.ShiftScaleRotate(
                    shift_limit=24. / 256, scale_limit=0.15, rotate_limit=30,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REPLICATE,
                    p=0.8),
            ])
            processed = augmentations(image=slices_image)
            slices_image = processed['image']

        img = torch.tensor(slices_image, dtype=torch.float32)

        if self.return_labels:
            labels = torch.tensor(self.data.loc[idx, ['epidural',
                                                      'intraparenchymal',
                                                      'intraventricular',
                                                      'subarachnoid',
                                                      'subdural',
                                                      'any']], dtype=torch.float32)
            return {'image': img, 'labels': labels, 'path': path}

        else:
            return {'image': img}
