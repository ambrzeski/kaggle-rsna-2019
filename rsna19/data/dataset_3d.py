from rsna19.data.dataset_2dc import IntracranialDataset

import numpy as np
import cv2


class IntracranialDataset3D(IntracranialDataset):
    _HU_AIR = -1000

    def __init__(self, config, folds, test=False, augment=False):
        """
        :param folds: list of selected folds
        :param return_labels: if True, labels will be returned with image
        """
        super().__init__(config, folds, test, augment)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        sample['image'] = sample['image'].unsqueeze(0)
        return sample


def main():
    from rsna19.configs.clf2Dc import Config

    for x in IntracranialDataset3D(Config(), folds=[0, 1, 2, 3], augment=True):
        print(x['image'].shape)
        for img_slice in list(x['image'][0]):
            img_slice = np.clip(((np.array(img_slice) + 1)*255), 0, 255).astype(np.uint8)
            cv2.imshow('slice', img_slice)
            cv2.waitKey()


if __name__ == "__main__":
    main()

