import json
from torch.nn import functional as F
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd

from rsna19.data.dataset_2dc import IntracranialDataset
from rsna19.models.clf2Dc.classifier2dc import Classifier2DC


def predict(checkpoint_path, device, subset):
    assert subset in ['train', 'val', 'test']
    train_dir = os.path.join(os.path.dirname(checkpoint_path), '..')
    config_path = os.path.join(train_dir, 'version_0/config.json')

    checkpoint_name = os.path.basename(checkpoint_path).split('.')[0]
    df_out_path = os.path.join(train_dir, f'results/{checkpoint_name}_{subset}.csv')
    os.makedirs(os.path.dirname(df_out_path), exist_ok=True)

    with open(config_path, 'r') as f:
        config = type('config', (), json.load(f))

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

        dataset = IntracranialDataset(config, folds, subset == 'test', False)

        all_paths = []
        all_study_id = []
        all_slice_num = []
        all_gt = []
        all_pred = []

        batch_size = 256
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


if __name__ == '__main__':
    checkpoint_paths = [
        '/kolos/m2/ct/models/classification/rsna/0009_regularization/0_1_2_3/models/_ckpt_epoch_3.ckpt',
        '/kolos/m2/ct/models/classification/rsna/0009_regularization/0_1_2_4/models/_ckpt_epoch_3.ckpt'
    ]

    predict(checkpoint_paths[0], 0, 'test')
