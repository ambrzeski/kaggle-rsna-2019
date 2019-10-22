from glob import glob
from pathlib import Path

import albumentations
import albumentations.pytorch
import cv2
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

from rsna19.data.utils import normalize_train, load_scan_2dc, draw_seg, load_seg_slice
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

        # todo use csv
        seg_ids = [path.split('/')[-2] for path in glob(f'{self.config.data_root}/train/*/Untitled.nii.gz')]
        data = data[data.path.apply(lambda x: x.split('/')[2]).isin(seg_ids)]

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

        if self.config.data_version != '3d':
            raise NotImplementedError

        # todo it would be better to have generic paths in csv and parameter specifying which data version to use
        meta_path = os.path.join(os.path.dirname(path), '../meta.json')
        seg_path = os.path.join(os.path.dirname(path), '../Untitled.nii.gz')
        path = path.replace('npy/', self.config.data_version + '/')

        middle_img_path = Path(path)

        middle_img_num = int(middle_img_path.stem)
        slices_indices = list(range(middle_img_num - self.config.num_slices // 2,
                                    middle_img_num + self.config.num_slices // 2 + 1))

        slices_image = load_scan_2dc(middle_img_path, slices_indices, self.config.pre_crop_size)
        seg = load_seg_slice(seg_path, meta_path, middle_img_num, self.config.pre_crop_size)

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

        # transforms.append(albumentations.pytorch.ToTensorV2())

        processed = albumentations.Compose(transforms)(image=slices_image, mask=seg)
        img = processed['image']
        seg = processed['mask']

        img = (img * 2) - 1
        img = torch.tensor(img.transpose((2, 0, 1)), dtype=torch.float32)


        out_seg = np.zeros((self.config.n_classes, seg.shape[0], seg.shape[1]), dtype=np.float32)
        for class_ in range(1, self.config.n_classes):
            out_seg[class_ - 1] = np.float32(seg == class_)
        # last class is any
        out_seg[-1] = np.float32(np.any(out_seg[:-1], axis=0))
        out_seg = torch.tensor(out_seg)

        out = {
            'image': img,
            'seg': out_seg,
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

    dataset = IntracranialDataset(config, [0, 1, 2, 3, 4], augment=True)
    show_all_slices = False
    draw_any = False

    for i in range(20):
        sample = dataset[i]
        img = sample['image'].numpy()
        seg = sample['seg'].numpy()
        print(sample['labels'], img.shape, img.min(), img.max())

        img = np.uint8((img + 1) * 127.5)

        indices = range(img.shape[0]) if show_all_slices else [img.shape[0] // 2]
        for j in indices:
            slice_ = img[j]
            if j == img.shape[0] // 2:
                slice_ = draw_seg(slice_, seg, draw_any)
            plt.imshow(cv2.cvtColor(slice_, cv2.COLOR_BGR2RGB))
            plt.show()
