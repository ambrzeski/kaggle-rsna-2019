import numpy as np
import os
import pandas as pd


def generate_submission(prediction_paths, out_path, clip_eps=0.0):
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
    prediction_paths = '''
TODO for stage 2: list all predictions here like:
/home/dmytro/devbox/ml/kaggle-rsna-2019/output/prediction/test/resnet34_400_5_planes_combine_last_var_dr0/fold_0_ch7_normal.csv
/home/dmytro/devbox/ml/kaggle-rsna-2019/output/prediction/test/resnet34_400_5_planes_combine_last_var_dr0/fold_1_ch7_normal.csv
/home/dmytro/devbox/ml/kaggle-rsna-2019/output/prediction/test/resnet34_400_5_planes_combine_last_var_dr0/fold_2_ch7_normal.csv
/home/dmytro/devbox/ml/kaggle-rsna-2019/output/prediction/test/resnet34_400_5_planes_combine_last_var_dr0/fold_3_ch7_normal.csv
/home/dmytro/devbox/ml/kaggle-rsna-2019/output/prediction/test/resnet34_400_5_planes_combine_last_var_dr0/fold_4_ch7_normal.csv
 '''.split()
    out_path = 'sub_stage2.csv'

    generate_submission(prediction_paths, out_path, clip_eps=1e-5)
