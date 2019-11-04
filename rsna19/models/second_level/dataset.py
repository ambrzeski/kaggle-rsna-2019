import os
from glob import glob
import numpy as np
import pandas as pd
from more_itertools import windowed, flatten
from rsna19.configs.second_level import Config
from sklearn.metrics import log_loss
from tqdm import tqdm


def main(config):
    train_x, train_y, val_x, val_y = create_dataset(config)

    os.makedirs(config.cache_dir, exist_ok=True)

    np.save(config.train_x, train_x)
    np.save(config.train_y, train_y)
    np.save(config.val_x, val_x)
    np.save(config.val_y, val_y)


def create_dataset(config):
    train_x_folds = []
    train_y_folds = []
    val_x_folds = []
    val_y_folds = []
    dfs = {}

    for model in config.models:
        tmp_dfs = []
        for path in glob(str(config.models_root / model / config.fold / "predictions" / "val_*.csv")):
            print(path)
            tmp_dfs.append(pd.read_csv(path))

        for i in range(1, len(tmp_dfs)):
            tmp_dfs[0][config.pred_columns] += tmp_dfs[i][config.pred_columns]

        tmp_dfs[0][config.pred_columns] /= len(tmp_dfs)
        dfs[model] = tmp_dfs[0]

    for i, (model, df) in enumerate(dfs.items()):
        print(model)
        df_areas = pd.read_csv(config.seg_areas_path)

        # Merge and standardize mask areas
        df = pd.merge(df, df_areas,  how='left', left_on=['study_id', 'slice_num'], right_on=['id', 'slice_number'])
        df.area = (df.area - df.area.mean()) / df.area.std()

        if i == 0:
            np.random.seed(9)
            study_ids = df.study_id.unique()
            np.random.shuffle(study_ids)

            train_ids = study_ids[:len(study_ids)//2]
            val_ids = study_ids[len(study_ids)//2:]

        train_df = df[df.study_id.isin(train_ids)]
        val_df = df[df.study_id.isin(val_ids)]

        train_gt = train_df[config.gt_columns].to_numpy()
        train_pred = train_df[config.pred_columns].to_numpy()
        val_gt = val_df[config.gt_columns].to_numpy()
        val_pred = val_df[config.pred_columns].to_numpy()

        train_log_loss = log_loss(train_gt.flatten(), train_pred.flatten(), sample_weight=config.class_weights * train_gt.shape[0])
        val_log_loss = log_loss(val_gt.flatten(), val_pred.flatten(), sample_weight=config.class_weights * val_gt.shape[0])
        print(f'{train_log_loss}, {val_log_loss}')

        train_x, train_y = create_split(train_df, train_ids, config)
        val_x, val_y = create_split(val_df, val_ids, config)

        train_x_folds.append(train_x)
        train_y_folds.append(train_y)
        val_x_folds.append(val_x)
        val_y_folds.append(val_y)

    train_x_folds = np.hstack(train_x_folds)
    train_y_folds = train_y_folds[0]
    val_x_folds = np.hstack(val_x_folds)
    val_y_folds = val_y_folds[0]

    return train_x_folds, train_y_folds, val_x_folds, val_y_folds


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

        if config.append_area_feature:
            study_areas = study_df['area'].to_numpy()
            study_areas = np.pad(study_areas, ((config.num_slices // 2, config.num_slices // 2),))
            study_areas = study_areas[new_indices].reshape(study_gt.shape[0], config.num_slices)
            study_x = np.concatenate((study_x, study_areas), axis=1)

        x.append(study_x)
        gt.append(study_gt)

    x = np.concatenate(x)
    gt = np.concatenate(gt)

    return x, gt


if __name__ == "__main__":
    main(Config())
