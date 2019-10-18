import numpy as np
import os
import pandas as pd


def generate_submission(prediction_paths, out_path, clip_eps=0):
    """
    Generate submission file by averaging predictions from multiple models read from csv files.
    Each csv file should contain predictions on test set from different model and must have following columns:
    [study_id, slice_num, pred_epidural, pred_intraparenchymal, pred_intraventricular, pred_subarachnoid, pred_subdural, pred_any]

    :param prediction_paths: path to csv files with predictions
    :param out_path: destination path of csv submission file
    :param clip_eps: eps used for clipping predictions to (eps, 1-eps) range
    """

    id_map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv/id_map.csv")
    id_map = pd.read_csv(id_map_path)

    pred_df = pd.concat([pd.read_csv(path) for path in prediction_paths])
    pred_df = pred_df.groupby(['study_id', 'slice_num']).mean()
    pred_df = pred_df.merge(id_map, on=['study_id', 'slice_num'])

    class_dfs = []
    classes = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
    for class_ in classes:
        df = pd.concat((pred_df['SOPInstanceUID'].apply(lambda x: f'{x}_{class_}'), pred_df[f'pred_{class_}']), axis=1)
        df = df.rename(columns={'SOPInstanceUID': 'ID', f'pred_{class_}': 'Label'})
        class_dfs.append(df)

    submission_df = pd.concat(class_dfs).sort_values(by='ID')
    if clip_eps > 0:
        submission_df.Label = np.clip(submission_df.Label, clip_eps, 1 - clip_eps)
    submission_df.to_csv(out_path, index=False, float_format='%.8f')


if __name__ == '__main__':
    prediction_paths = [
        # '/kolos/m2/ct/models/classification/rsna/0009_regularization/0_1_2_3/results/_ckpt_epoch_3_test.csv',
        # '/kolos/m2/ct/models/classification/rsna/0009_regularization/0_1_2_4/results/_ckpt_epoch_3_test.csv',
        # '/kolos/m2/ct/models/classification/rsna/0009_regularization/0_1_3_4/results/_ckpt_epoch_3_test.csv',
        # '/kolos/m2/ct/models/classification/rsna/0009_regularization/0_2_3_4/results/_ckpt_epoch_4_test.csv',
        # '/kolos/m2/ct/models/classification/rsna/0009_regularization/1_2_3_4/results/_ckpt_epoch_3_test.csv',
        # '/kolos/m2/ct/models/classification/rsna/0006_cdf2/0_1_2_3/results/_ckpt_epoch_2_test.csv',
        # '/kolos/m2/ct/models/classification/rsna/0006_cdf2/0_1_2_4/results/_ckpt_epoch_2_test.csv',
        # '/kolos/m2/ct/models/classification/rsna/0007_window1/0_1_2_3/results/_ckpt_epoch_2_test.csv',
        # '/kolos/m2/ct/models/classification/rsna/0008_window2/0_1_2_3/results/_ckpt_epoch_2_test.csv',
        '/kolos/m2/ct/models/classification/rsna/0014_384/0123/results/_ckpt_epoch_2_test.csv',
        '/kolos/m2/ct/models/classification/rsna/0014_384/0124/results/_ckpt_epoch_2_test.csv',
        '/kolos/m2/ct/models/classification/rsna/0014_384/0134/results/_ckpt_epoch_4_test.csv',
        '/kolos/m2/ct/models/classification/rsna/0014_384/0234/results/_ckpt_epoch_4_test.csv',
        '/kolos/m2/ct/models/classification/rsna/0014_384/1234/results/_ckpt_epoch_4_test.csv'
    ]
    out_path = '/kolos/m2/ct/data/rsna/submissions/submission9.csv'
    generate_submission(prediction_paths, out_path, 1e-6)
