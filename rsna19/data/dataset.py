import math
import os
from glob import glob

import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import torch
import skimage
from torch.utils.data import Dataset
from tqdm import tqdm

from preprocessing import hu_converter
from rsna19.configs.base_config import BaseConfig

from rsna19.data.utils import load_seg_slice, timeit_context, load_seg_3d


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
                 apply_windows=None,
                 add_segmentation_masks=False,
                 segmentation_oversample=20
                 ):
        """
        :param csv_file: path to csv file
        :param folds: list of selected folds
        :param csv_root_dir: prepended to csv_file path, defaults to project's rsna19/datasets
        :param return_labels: if True, labels will be returned with image
        :param preprocess_func: preprocessing function, e.g. for window adjustment
        """

        self.segmentation_oversample = segmentation_oversample
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
        study_ids = [path.split('/')[2] for path in data.path]
        data['study_id'] = study_ids

        if not is_test:
            data = data[data.fold.isin(folds)]

        if add_segmentation_masks:
            seg_ids = {path.split('/')[-2] for path in glob(f'{BaseConfig.data_root}/segmentation_masks/*/Untitled.nii.gz')}
        else:
            seg_ids = set()

        self.seg_ids = seg_ids.intersection(data['study_id'].unique())
        seg_data = data[data.study_id.isin(self.seg_ids)].copy()
        data = data[~data.study_id.isin(self.seg_ids)]

        seg_data = seg_data.reset_index()
        data = data.reset_index()

        self.data = data
        self.seg_data = seg_data

        if add_segmentation_masks:
            self.segmentation_masks = self.load_segmentation_masks()
        else:
            self.segmentation_masks = {}

        print(len(seg_ids))

    def load_segmentation_masks(self):
        res = {}
        for study_id in tqdm(self.seg_ids):
            meta_path = f'{BaseConfig.data_root}/segmentation_masks/{study_id}/meta.json'
            seg_path = f'{BaseConfig.data_root}/segmentation_masks/{study_id}/Untitled.nii.gz'
            seg = load_seg_3d(seg_path, meta_path)
            if seg.shape[1:] != (self.img_size, self.img_size):
                seg = np.array([
                    skimage.transform.resize(np.float32(seg[i]), (self.img_size, self.img_size),
                                             order=0, anti_aliasing=False).astype(seg.dtype)
                    for i in range(seg.shape[0])
                ])
            res[study_id] = seg

        return res

    def __len__(self):
        return len(self.seg_data) * self.segmentation_oversample + len(self.data)

    def __getitem__(self, idx):
        if idx < len(self.seg_data) * self.segmentation_oversample:
            dataset = self.seg_data
            dataset_idx = idx % len(self.seg_data)
            have_segmentation = True
        else:
            dataset = self.data
            dataset_idx = idx - len(self.seg_data) * self.segmentation_oversample
            have_segmentation = False

        data_path = dataset.loc[dataset_idx, 'path'].replace('/npy/', '/3d/')

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

        seg = None
        if have_segmentation:
            # meta_path = f'{BaseConfig.data_root}/segmentation_masks/{study_id}/meta.json'
            # seg_path = f'{BaseConfig.data_root}/segmentation_masks/{study_id}/Untitled.nii.gz'
            # with timeit_context('Load segmentation'):
            # seg = load_seg_slice(seg_path, meta_path, slice_num, self.img_size)
            seg = self.segmentation_masks[study_id][slice_num]

        if self.preprocess_func:
            if self.num_slices == 5:
                img_size = self.center_crop if self.center_crop > 0 else self.img_size
                img = np.concatenate([img,
                                      np.full((img_size, img_size, 1), self._HU_AIR, dtype=np.float)], axis=2)
                if have_segmentation:
                    processed = self.preprocess_func(image=img, mask=seg)
                    seg = processed['mask']
                else:
                    processed = self.preprocess_func(image=img)
                img = processed['image'][:, :, :-1]
            else:
                if have_segmentation:
                    processed = self.preprocess_func(image=img, mask=seg)
                    seg = processed['mask']
                else:
                    processed = self.preprocess_func(image=img)
                img = processed['image']

        out_seg = np.zeros((BaseConfig.n_classes+1, img.shape[0], img.shape[1]), dtype=np.float32)

        if have_segmentation:
            if self.center_crop > 0:
                from_row = (self.img_size - self.center_crop) // 2
                from_col = (self.img_size - self.center_crop) // 2
                seg = seg[from_row:from_row + self.center_crop, from_col:from_col + self.center_crop]

            for class_ in range(1, BaseConfig.n_classes+1):
                out_seg[class_ - 1] = np.float32(seg == class_)
            # last class is any
            out_seg[-1] = np.float32(np.any(out_seg[:-1], axis=0))

        out_seg = torch.tensor(out_seg)
        img = torch.from_numpy(img.transpose(2, 0, 1))

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
            'slice_num': slice_num,
            'seg': out_seg,
            'have_segmentation': have_segmentation
        }

        if self.return_labels:
            labels = torch.tensor(dataset.loc[
                                      dataset_idx,
                                      [
                                          'epidural',
                                          'intraparenchymal',
                                          'intraventricular',
                                          'subarachnoid',
                                          'subdural',
                                          'any'
                                      ]], dtype=torch.float)
            res['labels'] = labels

        return res




def print_stats(title, array):
    if len(array):
        print('{} shape:{} dtype:{} min:{} max:{} mean:{} median:{}'.format(
            title,
            array.shape,
            array.dtype,
            np.min(array),
            np.max(array),
            np.mean(array),
            np.median(array)
        ))
    else:
        print(title, 'empty')


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
                             add_segmentation_masks=True,
                             convert_cdf=True,
                             num_slices=5,
                             preprocess_func=albumentations.Compose([
                                 albumentations.ShiftScaleRotate(
                                     shift_limit=16./256, scale_limit=0.1, rotate_limit=30,
                                     interpolation=cv2.INTER_LINEAR,
                                     border_mode=cv2.BORDER_REPLICATE,
                                     p=0.99)
                             ]),
                             )
    sample = ds[2]
    img = sample['image']  # .detach().numpy()
    print_stats('images', img.detach().numpy())

    print(sample['labels'], img.shape, img.min(), img.max())
    for i in range(img.shape[0]):
        plt.imshow(img[i], cmap='gray')
        plt.show()

    seg = sample['seg']

    for i in range(seg.shape[0]):
        plt.imshow(seg[i], cmap='gray')
        plt.show()
