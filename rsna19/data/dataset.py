import os

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset

from rsna19.config import config


class IntracranialDataset(Dataset):
    def __init__(self,
                 csv_file,
                 folds,
                 csv_root_dir=None,
                 return_labels=True,
                 preprocess_func=None,
                 img_size=512,
                 scale_values=1.0):
        """
        :param csv_file: path to csv file
        :param folds: list of selected folds
        :param csv_root_dir: prepended to csv_file path, defaults to project's rsna19/datasets
        :param return_labels: if True, labels will be returned with image
        :param preprocess_func: preprocessing function, e.g. for window adjustment
        """

        self.return_labels = return_labels
        self.preprocess_func = preprocess_func
        self.img_size = img_size
        self.scale_values = scale_values  # scale all images data to values around 1

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
        img = np.load(img_path).astype(np.float) * self.scale_values
        s0, s1 = img.shape
        if s0 < 512:
            img = np.pad(img, ((0, 512 - s0), (0, 0)), mode='edge')
        if s0 > 512:
            img = img[:512, :]
        if s1 < 512:
            img = np.pad(img, ((0, 0), (0, 512-s1)), mode='edge')
        if s1 > 512:
            img = img[:, :512]

        if self.img_size != 512:
            img = cv2.resize(img, (self.img_size, self.img_size), cv2.INTER_AREA)

        img = img[None, :, :]

        if self.preprocess_func:
            img = self.preprocess_func(img)

        if self.return_labels:
            labels = torch.tensor(self.data.loc[idx, ['epidural',
                                                      'intraparenchymal',
                                                      'intraventricular',
                                                      'subarachnoid',
                                                      'subdural',
                                                      'any']], dtype=torch.float)
            return {'image': img, 'labels': labels}
        else:
            return {'image': img}


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ds = IntracranialDataset(csv_file='5fold.csv', folds=[0], img_size=256)
    sample = ds[0]
    print(sample['labels'], sample['image'].shape)
    plt.imshow(sample['image'][0])
    plt.show()
