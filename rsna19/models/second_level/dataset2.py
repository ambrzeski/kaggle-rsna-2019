import os
from glob import glob
import numpy as np
import pandas as pd
from more_itertools import windowed, flatten
from rsna19.configs.second_level import Config
from sklearn.metrics import log_loss
from tqdm import tqdm


def main(config):
    x, y = create_dataset(config, pred_type='val', stage='stage1')

    os.makedirs(config.cache_dir, exist_ok=True)

    np.save(config.cache_dir / "x.npy", x)
    np.save(config.cache_dir / "y.npy", y)

    # to make sure it does not fail as test prediction have no gt fields
    config.gt_columns = config.pred_columns
    x, y = create_dataset(config, pred_type='test', stage='stage2')

    np.save(config.cache_dir / "x_test.npy", x)


def create_dataset(config, pred_type, stage):
    x_folds = []
    y_folds = []
    dfs = {}

    for model in config.models:
        tmp_dfs = []
        for path in glob(str(config.models_root / model / config.fold / f"predictions" / f"{pred_type}_*.csv")):
            print(path)
            tmp_dfs.append(pd.read_csv(path))

        for i in range(1, len(tmp_dfs)):
            tmp_dfs[0][config.pred_columns] += tmp_dfs[i][config.pred_columns]

        tmp_dfs[0][config.pred_columns] /= len(tmp_dfs)
        dfs[model] = tmp_dfs[0]

    for i, (model, df) in enumerate(dfs.items()):
        print(model)
        df = df.sort_values(by=['study_id', 'slice_num'])
        study_ids = df.study_id.unique()

        gt = df[config.gt_columns].to_numpy()
        pred = df[config.pred_columns].to_numpy()

        if pred_type == 'val':
            init_log_loss = log_loss(gt.flatten(), pred.flatten(), sample_weight=config.class_weights * gt.shape[0])
            print(f'{init_log_loss}')

        x, y = create_split(df, study_ids, config)

        x_folds.append(x)
        y_folds.append(y)

    x_folds = np.hstack(x_folds)
    y_folds = y_folds[0]

    return x_folds, y_folds


def create_split(df, study_ids, config):
    x = []
    gt = []

    for study_id in tqdm(study_ids):
        # Get slices for current study_id
        study_df = df[df.study_id == study_id].sort_values('slice_num')

        study_preds = study_df[config.pred_columns].to_numpy()
        study_gt = study_df[config.gt_columns].to_numpy()

        study_preds = np.pad(study_preds, ((config.num_slices // 2, config.num_slices // 2), (0, 0)))

        new_indices = list(flatten(windowed(range(study_preds.shape[0]), config.num_slices)))
        study_x = study_preds[new_indices].reshape(study_gt.shape[0], config.predictions_in)

        x.append(study_x)
        gt.append(study_gt)

    x = np.concatenate(x)
    gt = np.concatenate(gt)

    return x, gt


if __name__ == "__main__":
    main(Config())
