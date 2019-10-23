import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations
import albumentations.pytorch
import cv2

from rsna19.data.utils import normalize_train
from rsna19.preprocessing.hu_converter import HuConverter


class IntracranialDataset(Dataset):
    _HU_AIR = -1000

    def __init__(self, config, folds, test=False, augment=False):
        """
        :param folds: list of selected folds
        :param return_labels: if True, labels will be returned with image
        """
        self.config = config
        self.test = test
        self.augment = augment

        if config.csv_root_dir is None:
            csv_root_dir = os.path.normpath(__file__ + '/../csv')
        else:
            csv_root_dir = config.csv_root_dir

        dataset_file = 'test.csv' if test else self.config.dataset_file
        data = pd.read_csv(os.path.join(csv_root_dir, dataset_file))
        if not test:
            data = data[data.fold.isin(folds)]
        data = data.reset_index()
        self.data = data

        if self.config.use_cdf:
            self.hu_converter = HuConverter

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data.loc[idx, 'path']
        study_id = path.split('/')[2]
        slice_num = os.path.basename(path).split('.')[0]
        path = os.path.normpath(os.path.join(self.config.data_root, '..', path))

        # todo it would be better to have generic paths in csv and parameter specifying which data version to use
        path = path.replace('npy/', self.config.data_version + '/')

        middle_img_path = Path(path)

        middle_img_num = int(middle_img_path.stem)
        slices_image = np.zeros((self.config.num_slices, self.config.pre_crop_size, self.config.pre_crop_size))
        for slice_idx, img_num in enumerate(range(middle_img_num - self.config.num_slices // 2,
                                                  middle_img_num + self.config.num_slices // 2 + 1)):

            if img_num < 0 or img_num > (len(os.listdir(middle_img_path.parent)) - 1):
                slice_img = np.full((self.config.pre_crop_size, self.config.pre_crop_size), self._HU_AIR)
            else:
                slice_img = np.load(middle_img_path.parent.joinpath('{:03d}.npy'.format(img_num)))

            if slice_img.shape != (self.config.pre_crop_size, self.config.pre_crop_size):
                slice_img = cv2.resize(np.int16(slice_img), (self.config.pre_crop_size, self.config.pre_crop_size),
                                       interpolation=cv2.INTER_AREA)

            slices_image[slice_idx] = slice_img

        if self.config.use_cdf:
            slices_image = self.hu_converter.convert(slices_image)
        else:
            slices_image = normalize_train(slices_image,
                                           self.config.min_hu_value,
                                           self.config.max_hu_value)

        slices_image = (slices_image.transpose((1, 2, 0)) + 1) / 2

        transforms = []
        if self.augment:
            if self.config.vertical_flip:
                transforms.append(albumentations.VerticalFlip(p=0.5))

            if self.config.pixel_augment:
                transforms.append(albumentations.RandomBrightnessContrast(0.2, 0.2, False, 0.8))

            if self.config.elastic_transform:
                transforms.append(albumentations.ElasticTransform(
                    alpha=20,
                    sigma=6,
                    alpha_affine=10,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5
                ))

            transforms.extend([
                albumentations.HorizontalFlip(p=0.5),
                albumentations.ShiftScaleRotate(
                    shift_limit=0, scale_limit=0.15, rotate_limit=30,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.9),
            ])

        if self.augment and self.config.random_crop:
            transforms.append(albumentations.RandomCrop(self.config.crop_size, self.config.crop_size))
        else:
            transforms.append(albumentations.CenterCrop(self.config.crop_size, self.config.crop_size))

        transforms.append(albumentations.pytorch.ToTensorV2())

        processed = albumentations.Compose(transforms)(image=slices_image)
        img = (processed['image'] * 2) - 1

        # img = torch.tensor(slices_image, dtype=torch.float32)

        out = {
            'image': img,
            'path': path,
            'study_id': study_id,
            'slice_num': slice_num
        }

        if not self.test:
            out['labels'] = torch.tensor(self.data.loc[idx, ['epidural',
                                                             'intraparenchymal',
                                                             'intraventricular',
                                                             'subarachnoid',
                                                             'subdural',
                                                             'any']], dtype=torch.float32)

        return out


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from rsna19.configs.se_resnext50_2dc import Config as config

    dataset = IntracranialDataset(config, [0], augment=True)
    show_all_slices = False

    for i in range(10):
        sample = dataset[0]
        img = sample['image'].numpy()
        print(sample['labels'], img.shape, img.min(), img.max())
        if show_all_slices:
            for slice_ in img:
                plt.imshow(slice_, cmap='gray')
                plt.show()
        else:
            plt.imshow(img[img.shape[0] // 2], cmap='gray')
            plt.show()
