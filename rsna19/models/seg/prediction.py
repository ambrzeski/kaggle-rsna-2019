import torch
import numpy as np
import cv2

from rsna19.data.dataset_seg import IntracranialDataset
from rsna19.models.seg.segmentation_model import SegmentationModel
from rsna19.configs.segmentation_config import Config
from rsna19.data.utils import draw_seg
from tqdm import tqdm


class Prediction:
    def __init__(self, config):
        print('Loading "{}"...'.format(config.eval_checkpoint_path))

        self.model = SegmentationModel(config)
        checkpoint = torch.load(config.eval_checkpoint_path, map_location='cuda:0')

        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

    def __call__(self, x):
        y = self.model.forward(x)
        return y


def main():
    config = Config()
    config.batch_size = 1

    prediction = Prediction(config)
    heatmap_dataset = IntracranialDataset(config, config.val_folds)

    for i, sample in tqdm(enumerate(heatmap_dataset), ):
        # Display positives only
        print(sample['path'])
        # if sample['seg'].cpu().numpy().sum() == 0.0:
        #     continue

        x = torch.unsqueeze(sample['image'], 0)
        y_batched = prediction(x).detach().cpu().numpy()

        img = np.uint8((sample['image'].numpy()[config.num_slices // 2] + 1) * 127.5)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        predictions = draw_seg(img, y_batched[0])
        labels = draw_seg(img, sample['seg'].numpy())

        drawing = np.concatenate((img_rgb, labels), axis=1)
        drawing = np.concatenate((drawing, predictions), axis=1)

        cv2.imwrite('/kolos/m2/ct/data/rsna/visualization/{}.png'.format(i), drawing)
        cv2.imshow('img', drawing)
        # cv2.waitKey()


if __name__ == "__main__":
    main()
