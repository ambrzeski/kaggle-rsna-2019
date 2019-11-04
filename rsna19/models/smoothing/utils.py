import pandas as pd
from sklearn.metrics import log_loss
from tqdm import tqdm


def smooth_predictions(src, dst, std=0.7, show_progress=False):
    df = pd.read_csv(src)
    pred_columns = [c for c in df.columns if 'pred_' in c]

    new_df = df.copy()

    unique_ids = df.study_id.unique()
    if show_progress:
        unique_ids = tqdm(unique_ids)

    for study_id in unique_ids:
        study_df = df[df.study_id == study_id].sort_values(by='slice_num')
        new_df.update(study_df[pred_columns].rolling(5, center=True, win_type='gaussian').mean(std=std))

    new_df.to_csv(dst, index=False)


def compute_log_loss(src):
    df = pd.read_csv(src)

    gt_columns = [c for c in df.columns if 'gt_' in c]
    pred_columns = [c for c in df.columns if 'pred_' in c]

    gt = df[gt_columns].to_numpy()
    pred = df[pred_columns].to_numpy()

    return log_loss(gt.flatten(), pred.flatten(), sample_weight=[1, 1, 1, 1, 1, 2] * gt.shape[0])


def compute_log_loss_per_class(src):
    df = pd.read_csv(src)

    gt_columns = [c for c in df.columns if 'gt_' in c]
    pred_columns = [c for c in df.columns if 'pred_' in c]

    gt = df[gt_columns].to_numpy()
    pred = df[pred_columns].to_numpy()

    losses = []
    gt = gt.flatten()
    pred = pred.flatten()

    for i in range(6):
        losses.append(log_loss(gt[i::6], pred[i::6]))

    return losses


def test_log_loss_per_class():
    from rsna19.configs.second_level import Config
    avg_loss = compute_log_loss(Config().prediction_paths[0])
    losses = compute_log_loss_per_class(Config().prediction_paths[0])
    losses.append(losses[-1])
    assert avg_loss == sum(losses) / len(losses)


if __name__ == "__main__":
    test_log_loss_per_class()
