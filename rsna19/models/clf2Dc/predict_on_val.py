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


def predict_on_val(checkpoint_path, device):
    train_dir = os.path.join(os.path.dirname(checkpoint_path), '..')
    config_path = os.path.join(train_dir, 'version_0/config.json')
    df_out_path = os.path.join(train_dir, 'version_0/val_results.csv')

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

        val_dataset = IntracranialDataset(config, config.val_folds)

        all_paths = []
        all_gt = []
        all_pred = []

        batch_size = 256
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
        for bix, batch in tqdm(enumerate(val_loader), total=len(val_dataset) // batch_size):
            x, y = batch['image'], batch['labels']
            paths = batch['path']
            y_hat = F.sigmoid(model(x.cuda()))

            all_paths.extend(paths)
            all_gt.append(y.numpy())
            all_pred.append(y_hat.cpu().numpy())

    all_gt = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)

    columns = ['gt_epidural', 'gt_intraparenchymal', 'gt_intraventricular', 'gt_subarachnoid', 'gt_subdural', 'gt_any',
               'pred_epidural', 'pred_intraparenchymal', 'pred_intraventricular', 'pred_subarachnoid', 'pred_subdural',
               'pred_any']
    df = pd.DataFrame(np.hstack((all_gt, all_pred)), columns=columns)
    df = pd.concat((pd.DataFrame(all_paths, columns=['path']), df), axis=1)

    df.to_csv(df_out_path)


if __name__ == '__main__':
    checkpoint_paths = [
        '/kolos/m2/ct/models/classification/rsna/0_1_2_3/models/_ckpt_epoch_1.ckpt',
        '/kolos/m2/ct/models/classification/rsna/0_1_2_4/models/_ckpt_epoch_1.ckpt',
        '/kolos/m2/ct/models/classification/rsna/0_1_3_4/models/_ckpt_epoch_1.ckpt',
        '/kolos/m2/ct/models/classification/rsna/0_2_3_4/models/_ckpt_epoch_1.ckpt',
        '/kolos/m2/ct/models/classification/rsna/1_2_3_4/models/_ckpt_epoch_1.ckpt'
    ]

    predict_on_val(checkpoint_paths[4], 3)
