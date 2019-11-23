import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd

from rsna19.data import dataset_3d_v2
from rsna19.configs.base_config import BaseConfig
from rsna19.models.clf3D.experiments_3d import MODELS
from rsna19.models.clf2D.train import build_model_str
import albumentations
import albumentations.pytorch
from rsna19.models.clf2D.predict import Rotate90

# import ttach as tta


def predict(model_name, fold, epoch, is_test, df_out_path, mode='normal', run=None):
    model_str = build_model_str(model_name, fold, run)
    model_info = MODELS[model_name]

    checkpoints_dir = f'{BaseConfig.checkpoints_dir}/{model_str}'
    print('\n', model_name, '\n')

    model = model_info.factory(**model_info.args)

    preprocess_func = []
    if 'h_flip' in mode:
        preprocess_func.append(albumentations.HorizontalFlip(always_apply=True))
    if 'v_flip' in mode:
        preprocess_func.append(albumentations.VerticalFlip(always_apply=True))
    if 'rot90' in mode:
        preprocess_func.append(Rotate90(always_apply=True))
    preprocess_func.append(albumentations.pytorch.ToTensorV2())

    dataset_valid = dataset_3d_v2.IntracranialDataset(
        csv_file='test2.csv' if is_test else '5fold.csv',
        folds=[fold],
        preprocess_func=albumentations.Compose(preprocess_func),
        return_labels=not is_test,
        is_test=is_test,
        return_all_slices=True,
        **{**model_info.dataset_args}
    )

    print(f'load {checkpoints_dir}/{epoch:03}.pt')
    checkpoint = torch.load(f'{checkpoints_dir}/{epoch:03}.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()
    model.eval()

    batch_size = 1  # always use batch size 1 as nb slices is variable and likely not to fit GPU
    data_loader = DataLoader(dataset_valid,
                             shuffle=False,
                             num_workers=8,
                             batch_size=batch_size)

    all_paths = []
    all_study_id = []
    all_slice_num = []
    all_gt = []
    all_pred = []

    data_iter = tqdm(enumerate(data_loader), total=len(data_loader))
    for iter_num, batch in data_iter:
        # if iter_num > 100:
        #     break
        with torch.set_grad_enabled(False):
            all_paths += batch['path']
            nb_slices = len(batch['path'])
            study_id = batch['study_id'][0]
            all_study_id += [study_id] * nb_slices
            all_slice_num += list(batch['slice_num'][0].cpu().numpy())

            y_hat = torch.sigmoid(model(batch['image'].float().cuda()))[0]
            all_pred.append(y_hat.detach().cpu().numpy())

            if not is_test:
                y = batch['labels'].detach().cpu().numpy()[0]
                all_gt.append(y)

    pred_columns = ['pred_epidural', 'pred_intraparenchymal', 'pred_intraventricular', 'pred_subarachnoid',
                    'pred_subdural', 'pred_any']
    gt_columns = ['gt_epidural', 'gt_intraparenchymal', 'gt_intraventricular', 'gt_subarachnoid', 'gt_subdural',
                  'gt_any']

    if is_test:
        all_pred = np.concatenate(all_pred)
        df = pd.DataFrame(all_pred, columns=pred_columns)
    else:
        all_pred = np.concatenate(all_pred)
        all_gt = np.concatenate(all_gt)
        df = pd.DataFrame(np.hstack((all_gt, all_pred)), columns=gt_columns + pred_columns)
        print(all_pred.shape, all_gt.shape)

        class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 2.0])
        print(F.binary_cross_entropy(torch.from_numpy(all_pred),
                                     torch.from_numpy(all_gt),
                                     class_weights.repeat(all_pred.shape[0], 1)))

    df = pd.concat((df, pd.DataFrame({
        'path': all_paths, 'study_id': all_study_id, 'slice_num': all_slice_num})), axis=1)
    df.to_csv(df_out_path, index=False)


def predict_test(model_name, fold, epoch, mode='normal', run=None):
    run_str = '' if not run else f'_{run}'
    prediction_dir = f'{BaseConfig.prediction_dir}/{model_name}{run_str}/fold{fold}/predictions/'
    os.makedirs(prediction_dir, exist_ok=True)
    df_out_path = f'{prediction_dir}/test_{mode}.csv'
    print(df_out_path)
    if os.path.exists(df_out_path):
        print('Skip existing', df_out_path)
    else:
        predict(model_name=model_name, fold=fold, epoch=epoch, is_test=True, df_out_path=df_out_path, mode=mode, run=run)


def predict_oof(model_name, fold, epoch, mode='normal', run=None):
    run_str = '' if not run else f'_{run}'
    prediction_dir = f'{BaseConfig.prediction_dir}/{model_name}{run_str}/fold{fold}/predictions/'
    os.makedirs(prediction_dir, exist_ok=True)
    df_out_path = f'{prediction_dir}/val_{mode}.csv'
    print(df_out_path)
    if os.path.exists(df_out_path):
        print('Skip existing', df_out_path)
    else:
        predict(model_name=model_name, fold=fold, epoch=epoch, is_test=False, df_out_path=df_out_path, mode=mode, run=run)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, default='check')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--run', type=str, default='')
    parser.add_argument('--fold', type=int, nargs='+')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--epoch', type=int, nargs='+')
    parser.add_argument('--mode', type=str, default=['normal'], nargs='+')

    parser.add_argument('--resume_weights', type=str, default='')
    parser.add_argument('--resume_epoch', type=int, default=-1)

    args = parser.parse_args()
    action = args.action
    modes = args.mode
    if modes == ['all']:
        modes = ['normal', 'h_flip', 'v_flip', 'rot90']

    if action == 'predict_test':
        for fold in args.fold:
            for epoch in args.epoch:
                for mode in modes:
                    print(f'fold {fold}, epoch {epoch}, {mode}')
                    predict_test(model_name=args.model, run=args.run, fold=fold, epoch=epoch, mode=mode)

    if action == 'predict_oof':
        for fold in args.fold:
            for epoch in args.epoch:
                for mode in modes:
                    print(f'fold {fold}, epoch {epoch}, {mode}')
                    predict_oof(model_name=args.model, run=args.run, fold=fold, epoch=epoch, mode=mode)
