import pandas as pd
import numpy as np
from rsna19.configs.base_config import BaseConfig
import scipy.ndimage
from tqdm import tqdm
import os
import glob
from multiprocessing import Pool
WORKERS = 8


def find_instance_centers(instance_dir):
    print(instance_dir)
    paths = glob.glob(f'{instance_dir}/npy/*.npy')
    # print(samples_dir, paths)
    samples = np.array([np.load(p) for p in paths])
    samples = ((samples > 0) & (samples < 80)).astype(np.float)
    samples_mean = np.mean(samples, axis=0)
    # print(samples_mean.shape)

    center_row, center_col = scipy.ndimage.measurements.center_of_mass(samples_mean)
    print(instance_dir, center_row, center_col)
    return instance_dir.split('/')[-1], int(center_row), int(center_col)

    # res.append([study_instance, int(center_row), int(center_col)])
    # print()


def find_centers(samples_dir):
    study_instances = list(os.listdir(samples_dir))
    paths = [f'{samples_dir}/{study_instance}' for study_instance in study_instances]

    with Pool(WORKERS) as p:
        res = list(tqdm(p.imap(find_instance_centers, paths), total=len(paths)))

    df = pd.DataFrame(res, columns=['study_id', 'center_row', 'center_col'])
    # df['study_id'] = study_instances

    return df

def main():
    # find_centers(BaseConfig.data_root + '/train').to_csv('data/csv/train_centers.csv', index=False)
    # find_centers(BaseConfig.data_root + '/test').to_csv('data/csv/test_centers.csv', index=False)

    find_instance_centers(BaseConfig.data_root + '/train/' + 'ID_b26cbdc518')

if __name__ == "__main__":
    main()
