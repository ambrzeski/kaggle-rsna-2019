import glob
import os
import traceback
from multiprocessing import Pool

import cv2
import numpy as np
import pydicom
import tqdm

from rsna19.data.utils import load_labels
from rsna19.preprocessing.pydicom_loader import PydicomLoader

STEP = 25
PATH = "/kolos/m2/ct/data/rsna/*/*/dicom/*"
CLASSES = {
    "epidural":         (255, 237, 0),
    "intraparenchymal": (212, 36, 0),
    "intraventricular": (173, 102, 108),
    "subarachnoid":     (0, 48, 114),
    "subdural":         (74, 87, 50)
}

loader = PydicomLoader()
labels = load_labels()


def draw_labels(path):
    try:
        img = loader.load(path)
        scan_id = pydicom.dcmread(path).SOPInstanceUID
        img = ((img.astype(np.float32) / np.iinfo(np.uint16).max) * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        counter = 0

        if 'train' in path:
            for c in CLASSES.keys():
                if labels.loc[scan_id][c]:
                    cv2.circle(img, (STEP, STEP // 2 + counter * STEP),
                               STEP // 2, CLASSES[c], -1)
                    cv2.putText(img, c, (2 * STEP, STEP // 2 + counter * STEP),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, CLASSES[c])
                    counter += 1

        dst_path = path.replace("dicom", "vis").replace('.dcm', '.png')
        dst_storage_path = dst_path.replace('m2', 'storage')

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        os.makedirs(os.path.dirname(dst_storage_path), exist_ok=True)

        # if any(labels.loc[scan_id]):
        #     cv2.imshow('img', img)
        #     cv2.waitKey()
        #
        cv2.imwrite(dst_storage_path, img)

        if os.path.exists(dst_path):
            os.remove(dst_path)

        os.symlink(dst_storage_path, dst_path)
    except:
        traceback.print_exc()


def main():
    paths = list(glob.glob(PATH))

    with Pool(30) as p:
        r = list(tqdm.tqdm(p.imap(draw_labels, paths), total=len(paths)))


if __name__ == "__main__":
    main()
