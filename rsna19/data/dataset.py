import math
import os

import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
from preprocessing import hu_converter
from rsna19.configs.base_config import BaseConfig


class IntracranialDataset(Dataset):
    _HU_AIR = -1000

    def __init__(self,
                 csv_file,
                 folds,
                 is_test=False,
                 csv_root_dir=None,
                 return_labels=True,
                 preprocess_func=None,
                 img_size=512,
                 center_crop=-1,
                 scale_values=1.0,
                 num_slices=1,
                 convert_cdf=False,
                 apply_windows=None
                 ):
        """
        :param csv_file: path to csv file
        :param folds: list of selected folds
        :param csv_root_dir: prepended to csv_file path, defaults to project's rsna19/datasets
        :param return_labels: if True, labels will be returned with image
        :param preprocess_func: preprocessing function, e.g. for window adjustment
        """

        self.num_slices = num_slices
        self.convert_cdf = convert_cdf
        self.center_crop = center_crop
        self.apply_windows = apply_windows
        self.return_labels = return_labels
        self.preprocess_func = preprocess_func
        self.img_size = img_size
        self.scale_values = scale_values  # scale all images data to values around 1
        self.is_test = is_test

        if csv_root_dir is None:
            csv_root_dir = os.path.normpath(__file__ + '/../csv')

        self.hu_converter = hu_converter.HuConverter

        data = pd.read_csv(os.path.join(csv_root_dir, csv_file))
        if not is_test:
            data = data[data.fold.isin(folds)]
        data = data.reset_index()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_path = self.data.loc[idx, 'path'].replace('/npy/', '/3d/')
        study_id = data_path.split('/')[-3]
        slice_num = int(os.path.basename(data_path).split('.')[0])
        full_path = os.path.normpath(os.path.join(BaseConfig.data_root, '..', data_path))
        middle_img_path = Path(full_path)

        def load_img(cur_slice_num):
            try:
                img_path = middle_img_path.parent.joinpath('{:03d}.npy'.format(cur_slice_num))
                img = np.load(img_path).astype(np.float) * self.scale_values
            except FileNotFoundError:
                img = np.full((self.img_size, self.img_size), self._HU_AIR, dtype=np.float)

            if img.shape != (self.img_size, self.img_size):
                img = cv2.resize(img, (self.img_size, self.img_size), cv2.INTER_AREA)

            if self.center_crop > 0:
                from_row = (self.img_size - self.center_crop) // 2
                from_col = (self.img_size - self.center_crop) // 2
                img = img[from_row:from_row + self.center_crop, from_col:from_col + self.center_crop]

            if self.convert_cdf:
                img = self.hu_converter.convert(img, use_cdf=True)

            img = img[:, :, None]

            return img

        if self.num_slices == 1:
            img = load_img(slice_num)
        else:
            steps = int(math.floor(self.num_slices/2.0))
            img = np.concatenate(
                [load_img(slice) for slice in range(slice_num - steps, slice_num + steps + 1)],
                axis=2
            )

        if self.preprocess_func:
            if self.num_slices == 5:
                img_size = self.center_crop if self.center_crop > 0 else self.img_size
                img = np.concatenate([img,
                                      np.full((img_size, img_size, 1), self._HU_AIR, dtype=np.float)], axis=2)
                processed = self.preprocess_func(image=img)
                img = processed['image'][:-1, :, :]
            else:
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
            'study_id': study_id,
            'slice_num': slice_num
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

    ds = IntracranialDataset(csv_file='5fold.csv', folds=[1],
                             img_size=400,
                             # center_crop=384,
                             convert_cdf=True,
                             num_slices=5,
                             preprocess_func=albumentations.Compose([
                                 albumentations.ShiftScaleRotate(
                                     shift_limit=16./256, scale_limit=0.1, rotate_limit=30,
                                     interpolation=cv2.INTER_LINEAR,
                                     border_mode=cv2.BORDER_REPLICATE,
                                     p=0.99),
                                 albumentations.pytorch.ToTensorV2()
                             ]),
                             )
    sample = ds[0]
    img = sample['image']  # .detach().numpy()
    print(sample['labels'], img.shape, img.min(), img.max())
    for i in range(img.shape[0]):
        plt.imshow(img[i], cmap='gray')
        plt.show()
