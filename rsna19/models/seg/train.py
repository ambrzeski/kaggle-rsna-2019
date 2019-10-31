import json

import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from test_tube import Experiment

from rsna19.models.seg.segmentation_model import SegmentationModel
from rsna19.configs.segmentation_config import Config


def main():
    config = Config()
    model = SegmentationModel(config)

    exp = Experiment(config.train_out_dir)

    with open(os.path.join(exp.log_dir, '../config.json'), 'w') as f:
        config_dict = {k: getattr(Config, k) for k in dir(Config) if not k.startswith('__')}
        json.dump(config_dict, f, indent=2)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(config.train_out_dir, "models"),
        save_best_only=True,
        verbose=True,
        monitor='val_iou_any',
        mode='max',
        prefix=''
    )

    trainer = Trainer(experiment=exp,
                      distributed_backend='dp',
                      max_nb_epochs=config.max_epoch,
                      checkpoint_callback=checkpoint_callback,
                      gpus=config.gpus,
                      nb_sanity_val_steps=10,
                      val_check_interval=1,
                      row_log_interval=10
                      )

    trainer.fit(model)


if __name__ == '__main__':
    main()
