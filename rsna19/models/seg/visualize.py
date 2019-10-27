import os
from pathlib import Path

from tqdm import tqdm
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader

from rsna19.data.dataset_seg import IntracranialDataset
from rsna19.models.seg.segmentation_model import SegmentationModel
from rsna19.configs.segmentation_config import Config
from rsna19.configs import load
from rsna19.data.utils import draw_seg, draw_labels


MODEL = '/kolos/m2/ct/models/classification/rsna/seg0001_ours/1234/version_2/models/_ckpt_epoch_35.ckpt'
# MODEL = '/kolos/m2/ct/models/classification/rsna/seg0001_ours/0234/version_2/models/_ckpt_epoch_48.ckpt'
# MODEL = '/kolos/m2/ct/models/classification/rsna/seg0001_ours/0134/version_2/models/_ckpt_epoch_60.ckpt'
# MODEL = '/kolos/m2/ct/models/classification/rsna/seg0001_ours/0124/version_2/models/_ckpt_epoch_27.ckpt'
# MODEL = '/kolos/m2/ct/models/classification/rsna/seg0001_ours/0123/version_2/models/_ckpt_epoch_52.ckpt'
GPU = 0
BATCH_SIZE = 100
OUT_DIR = '/kolos/m2/ct/data/rsna/seg_visualization/fold{folds}'


class PredictionModel:

    def __init__(self, model_path, gpu):
        print(f'Loading "{model_path}"...')

        # Model
        config_path = Path(model_path).parents[1] / "config.json"
        self.config = load(config_path)
        self.model = SegmentationModel(self.config)

        # Checkpoint
        device = torch.device(gpu)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(device)
        self.model.eval()

    def __call__(self, x):
        y = self.model.predict(x)
        return y


def main():
    # Prepare model
    prediction_model = PredictionModel(MODEL, GPU)

    # Prepare data
    config = Config()
    seg_dataset = IntracranialDataset(
        config=config,
        folds=prediction_model.model.val_folds)
    loader = DataLoader(
        dataset=seg_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4
    )

    counter = 0

    # Run prediction
    for batch in tqdm(loader):

        y_batched = prediction_model(batch['image'].to(torch.device(GPU))).detach().cpu().numpy()

        for i, img in enumerate(batch['image']):

            img = np.uint8((img.numpy()[config.num_slices // 2] + 1) * 127.5)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            draw_labels(img_rgb, batch['labels'][i] > 0.5)
            predictions = draw_seg(img, y_batched[i], draw_any=True)
            labels = draw_seg(img, batch['seg'][i].numpy())
            path = Path(batch['path'][i])

            drawing = np.concatenate((img_rgb, labels), axis=1)
            drawing = np.concatenate((drawing, predictions), axis=1)

            val_folds_str = "".join(map(str, prediction_model.model.val_folds))
            out_dir = OUT_DIR.format(folds=val_folds_str)
            save_dir = Path(out_dir)
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(str(save_dir/f'{counter:05d}_{path.parts[-3]}_{Path(path.parts[-1]).stem}.png'), drawing)
            # cv2.imshow('img', drawing)
            # cv2.waitKey()
            counter += 1


if __name__ == "__main__":
    main()
