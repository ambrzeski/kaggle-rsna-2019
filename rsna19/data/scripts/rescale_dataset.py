import glob
import traceback
from multiprocessing.pool import Pool

import cv2 as cv
import numpy as np
import os
import tqdm

from rsna19.configs.base_config import BaseConfig as config

WORKERS = 12


def convert_sample(path_in):
    try:
        path_out = path_in.replace('npy/', 'npy256/')
        dir_out = os.path.dirname(path_out)

        img = np.load(path_in)
        img = cv.resize(np.int16(img), (256, 256), cv.INTER_AREA)

        os.makedirs(dir_out, exist_ok=True)
        np.save(path_out, img)
    except:
        traceback.print_exc()
        print(path_in)


def main():
    paths = glob.glob(config.data_root + 'train/*/npy/*.npy') + glob.glob(config.data_root + 'test/*/npy/*.npy')

    with Pool(WORKERS) as p:
        r = list(tqdm.tqdm(p.imap(convert_sample, paths), total=len(paths)))


if __name__ == "__main__":
    main()
