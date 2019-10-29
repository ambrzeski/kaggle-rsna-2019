import math
import os
import pickle
import random

import numpy as np
import pandas as pd
from pathlib import Path
import collections
import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from preprocessing import hu_converter
from rsna19.configs.base_config import BaseConfig

SliceInfo = collections.namedtuple('SliceInfo', 'study_id slice_num path labels')


class IntracranialDataset(Dataset):
    _HU_AIR = -1000

    def __init__(self,
                 csv_file,
                 folds,
                 is_test=False,
                 csv_root_dir=None,
                 return_labels=True,
                 random_slice=True,
                 return_all_slices=False,
                 preprocess_func=None,
                 img_size=400,
                 center_crop=-1,
                 num_slices=16,
                 convert_cdf=True,
                 apply_windows=None,
                 combine_slices_padding=1
                 ):
        """
        :param csv_file: path to csv file
        :param folds: list of selected folds
        :param csv_root_dir: prepended to csv_file path, defaults to project's rsna19/datasets
        :param return_labels: if True, labels will be returned with image
        :param preprocess_func: preprocessing function, e.g. for window adjustment
        """

        self.return_all_slices = return_all_slices
        self.combine_slices_padding = combine_slices_padding
        self.random_slice = random_slice
        self.num_slices = num_slices
        self.convert_cdf = convert_cdf
        self.center_crop = center_crop
        self.apply_windows = apply_windows
        self.return_labels = return_labels
        self.preprocess_func = preprocess_func
        self.img_size = img_size
        self.is_test = is_test

        if csv_root_dir is None:
            csv_root_dir = os.path.normpath(__file__ + '/../csv')

        self.hu_converter = hu_converter.HuConverter
        self.study_slices = self.load_study_slices(csv_root_dir, csv_file, folds, is_test)
        self.study_ids = list(sorted(list(self.study_slices.keys())))

    def load_study_slices(self, csv_root_dir, csv_file, folds, is_test):
        data = pd.read_csv(os.path.join(csv_root_dir, csv_file))
        if not is_test:
            data = data[data.fold.isin(folds)]
        data = data.reset_index()

        study_slices = {}
        for row in tqdm(data.itertuples()):
            path_items = row.path.split('/')
            study_id = path_items[2]
            fn = path_items[-1]
            slice_num = int(fn.split('.')[0])
            path_items[3] = '3d'
            if self.return_labels:
                labels = np.array([row.epidural,
                                   row.intraparenchymal,
                                   row.intraventricular,
                                   row.subarachnoid,
                                   row.subdural,
                                   row.any], dtype=np.float)
            else:
                labels = np.zeros((6,), dtype=np.float)

            if study_id not in study_slices:
                study_slices[study_id] = {}

            study_slices[study_id][slice_num] = SliceInfo(
                study_id=study_id,
                path='/'.join(path_items),
                slice_num=slice_num,
                labels=labels
            )

        return study_slices

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        slices = self.study_slices[study_id]
        slices_in_study = len(slices)

        if self.return_all_slices:
            first_slice = -self.combine_slices_padding
            last_slice = slices_in_study + self.combine_slices_padding
        else:
            if self.random_slice and slices_in_study + 2*self.combine_slices_padding > self.num_slices:
                first_slice = random.randrange(-self.combine_slices_padding,
                                               slices_in_study-self.num_slices+self.combine_slices_padding)
                last_slice = first_slice+self.num_slices
            else:
                first_slice = (slices_in_study - self.num_slices) * 2 // 3
                last_slice = first_slice+self.num_slices  # slices_in_study

        all_images = []
        all_labels = []
        all_paths = []
        for slice_idx in range(first_slice, last_slice):
            if 0 <= slice_idx < slices_in_study:
                all_paths.append(slices[slice_idx].path)
                full_path = os.path.normpath(os.path.join(BaseConfig.data_root, '..', slices[slice_idx].path))
                img = np.load(full_path).astype(np.float32)
                labels = slices[slice_idx].labels.astype(np.float32)
            else:
                all_paths.append('')
                img = np.full((self.img_size, self.img_size), self._HU_AIR, dtype=np.float32)
                labels = np.zeros((6,), dtype=np.float32)

            if self.center_crop > 0:
                center_row = img.shape[0] // 2
                center_col = img.shape[1] // 2
                from_row = int(np.clip(center_row - self.center_crop // 2, 0, self.img_size - self.center_crop))
                from_col = int(np.clip(center_col - self.center_crop // 2, 0, self.img_size - self.center_crop))
                img = img[from_row:from_row + self.center_crop, from_col:from_col + self.center_crop]

            if self.convert_cdf:
                img = self.hu_converter.convert(img, use_cdf=True)

            img = img[:, :, None]
            all_images.append(img)
            all_labels.append(labels)

        all_images = np.concatenate(all_images, axis=2)

        if self.preprocess_func:
            processed = self.preprocess_func(image=all_images)
            all_images = processed['image']

        all_labels = np.row_stack(all_labels)[self.combine_slices_padding:-self.combine_slices_padding]
        all_paths = all_paths[self.combine_slices_padding:-self.combine_slices_padding]

        # TODO: re-implement if necessary
        # if self.apply_windows is not None:
        #     if isinstance(all_images, torch.Tensor):
        #         slices = [
        #             torch.clamp(
        #                 (all_images - w_min) / (w_max - w_min),
        #                 0.0, 1.0
        #             ) for w_min, w_max in self.apply_windows
        #         ]
        #         all_images = torch.cat(slices, dim=0)
        #     else:
        #         slices = [
        #             np.clip(
        #                 (all_images - w_min) / (w_max - w_min),
        #                 0.0, 1.0
        #             ) for w_min, w_max in self.apply_windows
        #         ]
        #         all_images = np.concatenate(slices, axis=0)

        res = {
            # 'idx': np.array([idx] * len(all_labels)),
            'image': all_images,
            'study_id': study_id,
            'first_slice': first_slice,
            'slice_num': np.arange(first_slice+self.combine_slices_padding, last_slice-self.combine_slices_padding),
            'labels': all_labels,
            'path': all_paths
        }

        return res


def check_performance():
    import matplotlib.pyplot as plt
    import albumentations
    import albumentations.pytorch
    import cv2

    ds = IntracranialDataset(csv_file='5fold.csv', folds=[1],
                             img_size=400,
                             # center_crop=384,
                             convert_cdf=True,
                             num_slices=16,
                             random_slice=True,
                             preprocess_func=albumentations.Compose([
                                 albumentations.ShiftScaleRotate(shift_limit=16. / 256, scale_limit=0.1,
                                                                 rotate_limit=30,
                                                                 interpolation=cv2.INTER_LINEAR,
                                                                 border_mode=cv2.BORDER_REPLICATE,
                                                                 p=0.80),
                                 albumentations.Flip(),
                                 albumentations.RandomRotate90(),
                                 albumentations.pytorch.ToTensorV2()
                             ]),
                             )

    dl = torch.utils.data.DataLoader(ds,
               num_workers=0,
               shuffle=True,
               batch_size=1)

    for data in tqdm(dl):
        pass
        img = data['image'].float().cuda()
        labels = data['labels'] # .float().cuda()
        # print(img.shape)
        img = img[0, :, None, :, :]
        # print(img.shape)


def check_dataset():
    import matplotlib.pyplot as plt
    import albumentations
    import albumentations.pytorch
    import cv2

    def _w(w, l):
        return l - w / 2, l + w / 2

    ds = IntracranialDataset(csv_file='5fold.csv', folds=[1],
                             img_size=512,
                             # center_crop=384,
                             convert_cdf=True,
                             num_slices=4,
                             random_slice=True,
                             preprocess_func=albumentations.Compose([
                                 albumentations.ShiftScaleRotate(shift_limit=16. / 256, scale_limit=0.1,
                                                                 rotate_limit=30,
                                                                 interpolation=cv2.INTER_LINEAR,
                                                                 border_mode=cv2.BORDER_REPLICATE,
                                                                 p=0.80),
                                 albumentations.Flip(),
                                 albumentations.RandomRotate90(),
                                 albumentations.pytorch.ToTensorV2()
                             ]),
                             )
    sample = ds[0]
    img = sample['image'].detach().numpy()
    print(sample['labels'], img.shape, img.shape, img.min(), img.max(), sample['first_slice'], sample['study_id'])
    for i in range(img.shape[0]):
        plt.imshow(img[i], cmap='gray')
        plt.show()

    dl = torch.utils.data.DataLoader(ds,
                                     num_workers=0,
                                     shuffle=True,
                                     batch_size=1)

    for sample in tqdm(dl):
        img = sample['image'].detach().numpy()
        print(sample['labels'], img.shape, img.shape, img.min(), img.max(), sample['first_slice'], sample['study_id'])
        for i in range(img.shape[1]):
            plt.imshow(img[0, i], cmap='gray')
            plt.show()
        break

if __name__ == '__main__':
    # check_performance()
    check_dataset()
