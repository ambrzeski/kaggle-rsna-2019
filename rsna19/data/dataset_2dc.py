import os
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations
import albumentations.pytorch
import cv2

from rsna19.data.utils import normalize_train, load_scan_2dc, load_seg_masks_2dc
from rsna19.preprocessing.hu_converter import HuConverter


class IntracranialDataset(Dataset):
    _HU_AIR = -1000

    def __init__(self, config, folds, mode='train', augment=False, use_cq500=False, transforms=None):
        """
        :param folds: list of selected folds
        :param mode: 'train', 'val' or 'test'
        :param return_labels: if True, labels will be returned with image
        """
        self.config = config
        self.mode = mode
        self.augment = augment
        self.additional_transforms = transforms

        if config.csv_root_dir is None:
            csv_root_dir = os.path.normpath(__file__ + '/../csv')
        else:
            csv_root_dir = config.csv_root_dir

        dataset_file = None
        if self.mode == 'train':
            dataset_file = self.config.train_dataset_file
        if self.mode == 'val':
            dataset_file = self.config.val_dataset_file
        if self.mode == 'test':
            dataset_file = self.config.test_dataset_file

        data = pd.read_csv(os.path.join(csv_root_dir, dataset_file))

        if not mode == 'test':
            data = data[data.fold.isin(folds)]

        # protect from adding cq500 to validation
        if use_cq500:
            data_cq500 = pd.read_csv(os.path.join(csv_root_dir, 'cq500_5fold_cleared.csv'))
            data = pd.concat([data, data_cq500], axis=0)

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
        slices_indices = list(range(middle_img_num - self.config.num_slices // 2,
                                    middle_img_num + self.config.num_slices // 2 + 1))

        slices_image = load_scan_2dc(middle_img_path, slices_indices, self.config.pre_crop_size,
                                     self.config.padded_size)

        if self.config.use_cdf:
            slices_image = self.hu_converter.convert(slices_image)
        else:
            slices_image = normalize_train(slices_image,
                                           self.config.min_hu_value,
                                           self.config.max_hu_value)

        slices_image = (slices_image.transpose((1, 2, 0)) + 1) / 2

        # Load and append segmentation masks
        if hasattr(self.config, 'append_masks') and self.config.append_masks:
            seg_masks = load_seg_masks_2dc(middle_img_path, slices_indices, self.config.pre_crop_size)
            seg_masks = seg_masks.transpose((1, 2, 0))
            slices_image = np.concatenate((slices_image, seg_masks), axis=2)

        transforms = []
        if self.additional_transforms is not None:
            transforms.extend(self.additional_transforms)

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
                    shift_limit=self.config.shift_limit, scale_limit=0.15, rotate_limit=30,
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

        if not self.mode == 'test':
            out['labels'] = torch.tensor(self.data.loc[idx, ['epidural',
                                                             'intraparenchymal',
                                                             'intraventricular',
                                                             'subarachnoid',
                                                             'subdural',
                                                             'any']], dtype=torch.float32)

        return out


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from rsna19.configs.clf2Dc import Config as config

    dataset = IntracranialDataset(config, [0], augment=True)
    show_all_slices = False

    for i in range(10):
        sample = dataset[0]
        img = sample['image'].numpy()
        print(sample['labels'], img.shape, img.min(), img.max())
        if show_all_slices:
            for slice_ in img:
                plt.imshow(slice_, cmap='gray', vmin=-1, vmax=1)
                plt.show()
        else:
            plt.imshow(img[img.shape[0] // 2], cmap='gray')
            plt.show()
