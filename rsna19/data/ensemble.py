import glob

import pandas as pd
from tqdm import tqdm

from rsna19.configs.base_config import BaseConfig


PREDICTIONS_QUERY = BaseConfig.model_outdir + '/*/*/predictions_stage2'
SMOOTHEN = False
OUT = 'ensemble.csv'


def main():
    predictions = []
    prediction_dirs = glob.glob(PREDICTIONS_QUERY)
    for p in prediction_dirs:

        test_predictions = glob.glob(p + '/test*.csv')
        if test_predictions:
            print("Averaging: ", test_predictions)
            dfs = [pd.read_csv(path) for path in test_predictions]
            avg_df = average(dfs)

            if SMOOTHEN:
                print("Smoothing: ", test_predictions)
                df = smoothen(avg_df)
                df = average([avg_df, df])
            else:
                df = avg_df

            predictions.append(df)

    final = average(predictions)
    final.to_csv(OUT, index=False)


def average(dfs):
    df = pd.concat(dfs)
    df = df.groupby(['study_id', 'slice_num']).mean().reset_index()
    return df


def smoothen(df):
    pred_columns = [c for c in df.columns if 'pred_' in c]
    new_df = df.copy()
    for study_id in tqdm(df.study_id.unique()):
        study_df = df[df.study_id == study_id].sort_values(by='slice_num')
        new_df.update(study_df[pred_columns].rolling(5, center=True, win_type='gaussian').mean(std=0.7))
    return new_df


if __name__ == '__main__':
    main()
