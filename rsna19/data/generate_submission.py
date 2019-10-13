import os
import pandas as pd


def generate_submission(prediction_paths, out_path):
    """
    Generate submission file by averaging predictions from multiple models read from csv files.
    Each csv file should contain predictions on test set from different model and must have following columns:
    [study_id, slice_num, pred_epidural, pred_intraparenchymal, pred_intraventricular, pred_subarachnoid, pred_subdural, pred_any]

    :param prediction_paths: path to csv files with predictions
    :param out_path: destination path of csv submission file
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
    submission_df.to_csv(out_path, index=False, float_format='%.6f')


if __name__ == '__main__':
    prediction_paths = [
        '/kolos/m2/ct/models/classification/rsna/0009_regularization/0_1_2_3/results/_ckpt_epoch_3_test.csv',
        '/kolos/m2/ct/models/classification/rsna/0009_regularization/0_1_2_4/results/_ckpt_epoch_3_test.csv',
    ]
    out_path = '/kolos/m2/ct/data/rsna/submissions/submission1.csv'
    generate_submission(prediction_paths, out_path)
