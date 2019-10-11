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
            csv_root_dir = os.path.normpath(__file__ + '/../csv')

        data = pd.read_csv(os.path.join(csv_root_dir, csv_file))
        data = data[data.fold.isin(folds)]
        data = data.reset_index()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_path = self.data.loc[idx, 'path']
        img_path = os.path.normpath(os.path.join(config.data_root, '..', data_path))
        img = np.load(img_path).astype(np.float) * self.scale_values

        if img.shape != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size), cv2.INTER_AREA)

        img = img[:, :, None]

        if self.preprocess_func:
            processed = self.preprocess_func(image=img)
            img = processed['image']

        res = {
            'idx': idx,
            'image': img,
            'path': data_path,
        }

        if self.return_labels:
            labels = torch.tensor(self.data.loc[idx, ['epidural',
                                                      'intraparenchymal',
                                                      'intraventricular',
                                                      'subarachnoid',
                                                      'subdural',
                                                      'any']], dtype=torch.float)
            res['labels'] = labels

        return res


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import albumentations
    import albumentations.pytorch
    import cv2

    ds = IntracranialDataset(csv_file='5fold.csv', folds=[0], img_size=256,
                             preprocess_func=albumentations.Compose([
                                 albumentations.ShiftScaleRotate(
                                     shift_limit=16./256, scale_limit=0.1, rotate_limit=30,
                                     interpolation=cv2.INTER_LINEAR,
                                     border_mode=cv2.BORDER_REPLICATE,
                                     p=0.75),
                                 albumentations.pytorch.ToTensorV2()
                             ]),
                             )
    for i in range(16):
        sample = ds[0]
        img = sample['image']  # .detach().numpy()
        print(sample['labels'], img.shape, img.min(), img.max(), sample)
        plt.imshow(img[0])
        plt.show()
