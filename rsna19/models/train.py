import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from test_tube import Experiment

from rsna19.models.classifier2dc import Classifier2DC
from rsna19.configs.se_resnext50_2dc import Config


def main():
    config = Config()
    model = Classifier2DC(config)

    exp = Experiment(config.train_out_dir)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(config.train_out_dir, "models"),
        save_best_only=False,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )

    trainer = Trainer(experiment=exp,
                      distributed_backend='dp',
                      max_nb_epochs=config.max_epoch,
                      checkpoint_callback=checkpoint_callback,
                      gpus=config.gpus)

    trainer.fit(model)


if __name__ == '__main__':
    main()
