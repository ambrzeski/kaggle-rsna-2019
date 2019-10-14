import os

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset

from rsna19.configs.base_config import BaseConfig


class IntracranialDataset(Dataset):
    def __init__(self,
                 csv_file,
                 folds,
                 csv_root_dir=None,
                 return_labels=True,
                 preprocess_func=None,
                 img_size=512,
                 center_crop=-1,
                 scale_values=1.0,
                 apply_windows=None
                 ):
        """
        :param csv_file: path to csv file
        :param folds: list of selected folds
        :param csv_root_dir: prepended to csv_file path, defaults to project's rsna19/datasets
        :param return_labels: if True, labels will be returned with image
        :param preprocess_func: preprocessing function, e.g. for window adjustment
        """

        self.center_crop = center_crop
        self.apply_windows = apply_windows
        self.return_labels = return_labels
        self.preprocess_func = preprocess_func
        self.img_size = img_size
        self.scale_values = scale_values  # scale all images data to values around 1

        if csv_root_dir is None:
            csv_root_dir = os.path.normpath(__file__ + '/../csv')

        if 'test' in csv_file:
            centers_csv_file = os.path.join(csv_root_dir, 'test_centers.csv')
        else:
            centers_csv_file = os.path.join(csv_root_dir, 'train_centers.csv')
        self.centers_data = pd.read_csv(centers_csv_file).set_index('study_id', drop=True)

        data = pd.read_csv(os.path.join(csv_root_dir, csv_file))
        data = data[data.fold.isin(folds)]
        data = data.reset_index()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_path = self.data.loc[idx, 'path']
        img_path = os.path.normpath(os.path.join(BaseConfig.data_root, '..', data_path))
        img = np.load(img_path).astype(np.float) * self.scale_values

        if img.shape != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size), cv2.INTER_AREA)

        if self.center_crop > 0:
            study_id = data_path.split('/')[-3]
            center_row, center_col = self.centers_data.loc[study_id, ['center_row', 'center_col']]
            from_row = int(np.clip(center_row - self.center_crop // 2, 0, self.img_size - self.center_crop))
            from_col = int(np.clip(center_col - self.center_crop // 2, 0, self.img_size - self.center_crop))

            img = img[from_row:from_row + self.center_crop, from_col:from_col + self.center_crop]

        img = img[:, :, None]

        if self.preprocess_func:
            processed = self.preprocess_func(image=img)
            img = processed['image']

        if self.apply_windows is not None:
            if isinstance(img, torch.Tensor):
                slices = [
                    torch.clamp(
                        (img - w_min) / (w_max - w_min),
                        0.0, 1.0
                    ) for w_min, w_max in self.apply_windows
                ]
                img = torch.cat(slices, dim=0)
            else:
                slices = [
                    np.clip(
                        (img - w_min) / (w_max - w_min),
                        0.0, 1.0
                    ) for w_min, w_max in self.apply_windows
                ]
                img = np.concatenate(slices, axis=0)

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

    def _w(w, l):
        return l - w / 2, l + w / 2

    ds = IntracranialDataset(csv_file='5fold.csv', folds=[0],
                             img_size=512,
                             center_crop=384,
                             apply_windows=[
                                _w(w=80, l=40),
                                _w(w=130, l=75),
                                _w(w=300, l=75),
                                _w(w=400, l=40),
                                _w(w=2800, l=600),
                                _w(w=8, l=32),
                                _w(w=40, l=40)
                             ],
                             preprocess_func=albumentations.Compose([
                                 # albumentations.ShiftScaleRotate(
                                 #     shift_limit=16./256, scale_limit=0.1, rotate_limit=30,
                                 #     interpolation=cv2.INTER_LINEAR,
                                 #     border_mode=cv2.BORDER_REPLICATE,
                                 #     p=0.99),
                                 albumentations.pytorch.ToTensorV2()
                             ]),
                             )
    sample = ds[0]
    img = sample['image']  # .detach().numpy()
    print(sample['labels'], img.shape, img.min(), img.max())
    for i in range(7):
        plt.imshow(img[i], cmap='gray')
        plt.show()
