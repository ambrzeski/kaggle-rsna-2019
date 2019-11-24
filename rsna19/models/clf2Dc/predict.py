import glob
import json
from pathlib import Path

import albumentations
import cv2
import numpy as np
import os
import pandas as pd
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from rsna19.configs.base_config import BaseConfig
from rsna19.data.dataset_2dc import IntracranialDataset
from rsna19.models.clf2Dc.classifier2dc import Classifier2DC


VAL_SET = '5fold.csv'
TEST_SET = 'test2.csv'


rot_params = {'interpolation': cv2.INTER_LINEAR, 'border_mode': cv2.BORDER_CONSTANT, 'value': 0, 'always_apply': True}

TTA_TRANSFORMS_RESNET34_3x3 = {
    # None: [albumentations.CenterCrop(384, 384)],
    'rcrop+rrot': [
        albumentations.Rotate((15, 15), **rot_params),
        albumentations.RandomCrop(384, 384)],
    'rcrop+lrot': [
        albumentations.Rotate((-15, -15), **rot_params),
        albumentations.RandomCrop(384, 384)],
    'rcrop+rrot+hflip': [
        albumentations.HorizontalFlip(True),
        albumentations.Rotate((15, 15), **rot_params),
        albumentations.RandomCrop(384, 384)],
    'rcrop+lrot+hflip': [
        albumentations.HorizontalFlip(True),
        albumentations.Rotate((-15, -15), **rot_params),
        albumentations.RandomCrop(384, 384)],
}

ssr_params = {'shift_limit': 0.1, 'scale_limit': 0.05, 'rotate_limit': 15, 'interpolation': cv2.INTER_LINEAR,
              'border_mode': cv2.BORDER_CONSTANT, 'value': 0, 'p': 0.9}

TTA_TRANSFORMS_RESNET50_7c_400 = {
    # None:   None,
    'ssr_1': [albumentations.ShiftScaleRotate(**ssr_params)],
    'ssr_2': [albumentations.ShiftScaleRotate(**ssr_params)],
    'ssr+hflip_1': [albumentations.HorizontalFlip(True), albumentations.ShiftScaleRotate(**ssr_params)],
    'ssr+hflip_2': [albumentations.HorizontalFlip(True), albumentations.ShiftScaleRotate(**ssr_params)],
}


def predict(checkpoint_path, device, subset, tta_transforms, tta_variant=None):
    assert subset in ['train', 'val', 'test']
    assert tta_variant in tta_transforms

    train_dir = os.path.join(os.path.dirname(checkpoint_path), '..')
    config_path = os.path.join(train_dir, 'version_0/config.json')

    if tta_variant is None:
        df_out_path = os.path.join(train_dir, f'predictions/{subset}_normal.csv')
    else:
        df_out_path = os.path.join(train_dir, f'predictions/{subset}_{tta_variant}.csv')
    if Path(df_out_path).exists():
        print(f'Predictions {df_out_path} already exist. Skipping.')
        return
    os.makedirs(os.path.dirname(df_out_path), exist_ok=True)

    with open(config_path, 'r') as f:
        config_dict = json.load(f)
        if 'dropout' not in config_dict:
            config_dict['dropout'] = 0
        if 'padded_size' not in config_dict:
            config_dict['padded_size'] = None
        if 'append_masks' not in config_dict:
            config_dict['append_masks'] = False
        if 'dataset_file' in config_dict:
            config_dict['train_dataset_file'] = config_dict['dataset_file']
        config_dict['val_dataset_file'] = VAL_SET
        config_dict['test_dataset_file'] = TEST_SET
        config = type('config', (), config_dict)

    with torch.cuda.device(device):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))

        model = Classifier2DC(config)
        model.load_state_dict(checkpoint['state_dict'])
        model.on_load_checkpoint(checkpoint)
        model.cuda()

        model.eval()
        model.freeze()

        if subset == 'train':
            folds = config.train_folds
        elif subset == 'val':
            folds = config.val_folds
        else:
            folds = None

        dataset = IntracranialDataset(config, folds, mode=subset, augment=False, transforms=tta_transforms[tta_variant])

        all_paths = []
        all_study_id = []
        all_slice_num = []
        all_gt = []
        all_pred = []

        batch_size = 128
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
        for bix, batch in tqdm(enumerate(data_loader), total=len(dataset) // batch_size):
            y_hat = F.sigmoid(model(batch['image'].cuda()))
            all_pred.append(y_hat.cpu().numpy())
            all_paths.extend(batch['path'])
            all_study_id.extend(batch['study_id'])
            all_slice_num.extend(batch['slice_num'])

            if subset != 'test':
                y = batch['labels']
                all_gt.append(y.numpy())

    pred_columns = ['pred_epidural', 'pred_intraparenchymal', 'pred_intraventricular', 'pred_subarachnoid',
                    'pred_subdural', 'pred_any']
    gt_columns = ['gt_epidural', 'gt_intraparenchymal', 'gt_intraventricular', 'gt_subarachnoid', 'gt_subdural',
                  'gt_any']

    if subset == 'test':
        all_pred = np.concatenate(all_pred)
        df = pd.DataFrame(all_pred, columns=pred_columns)
    else:
        all_pred = np.concatenate(all_pred)
        all_gt = np.concatenate(all_gt)
        df = pd.DataFrame(np.hstack((all_gt, all_pred)), columns=gt_columns + pred_columns)

    df = pd.concat((df, pd.DataFrame({
        'path': all_paths, 'study_id': all_study_id, 'slice_num': all_slice_num})), axis=1)
    df.to_csv(df_out_path, index=False)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':

    gpu = 0
    jobs = [
        (BaseConfig.model_outdir + '/0038_7s_res50_400', 'val', TTA_TRANSFORMS_RESNET50_7c_400),
        (BaseConfig.model_outdir + '/0038_7s_res50_400', 'test', TTA_TRANSFORMS_RESNET50_7c_400),
        (BaseConfig.model_outdir + '/0036_3x3_pretrained', 'val', TTA_TRANSFORMS_RESNET34_3x3),
        (BaseConfig.model_outdir + '/0036_3x3_pretrained', 'test', TTA_TRANSFORMS_RESNET34_3x3),
        (BaseConfig.model_outdir + '/0036_3x3_pretrained_stage2', 'val', TTA_TRANSFORMS_RESNET34_3x3),
        (BaseConfig.model_outdir + '/0036_3x3_pretrained_stage2', 'test', TTA_TRANSFORMS_RESNET34_3x3),
        (BaseConfig.model_outdir + '/0036_3x3_5_slices_pretrained', 'val', TTA_TRANSFORMS_RESNET34_3x3),
        (BaseConfig.model_outdir + '/0036_3x3_5_slices_pretrained', 'test', TTA_TRANSFORMS_RESNET34_3x3)
    ]

    for model_home, subset, ttas in jobs:

        model_dirs = glob.glob(model_home + "/*")
        for model_dir in model_dirs:

            checkpoint_path = sorted(glob.glob(model_dir + "/models/*.ckpt"))[-1]

            for tta in ttas:

                print(f"Calculating predictions for {checkpoint_path}, TTA: {tta}, subset: {subset}")
                predict(checkpoint_path, gpu, subset, ttas, tta)
