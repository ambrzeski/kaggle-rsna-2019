import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

from rsna19.data.dataset_seg import IntracranialDataset
from rsna19.models.commons.get_base_model import load_base_weights
from rsna19.models.commons.radam import RAdam


class SegmentationModel(pl.LightningModule):

    def __init__(self, config):
        super(SegmentationModel, self).__init__()
        self.config = config

        self.train_folds = config.train_folds
        self.val_folds = config.val_folds

        if config.pretrained == 'imagenet':
            self.model = smp.Unet(config.backbone, classes=config.n_classes, activation='sigmoid')
        else:
            self.model = smp.Unet(config.backbone, classes=config.n_classes, activation='sigmoid', encoder_weights=None)

            if config.pretrained is not None:
                weights = load_base_weights(config.pretrained, 3, '0.conv1.weight')
                weights = {'layer' + k: v for k, v in weights.items()}
                weights['last_linear.bias'] = None
                weights['last_linear.weight'] = None
                self.model.encoder.load_state_dict(weights)

        self.scheduler = None
        self.loss_func = smp.utils.losses.BCEDiceLoss(eps=1.)

        self.iou_metric = smp.utils.metrics.IoUMetric(eps=1., activation='sigmoid')
        self.f_score_metric = smp.utils.metrics.FscoreMetric(eps=1., activation='sigmoid')

    def forward(self, x):
        x = self.model(x)
        return x

    def predict(self, x):
        x = self.model.predict(x)
        return x

    # training step and validation step should return tensor or nested dicts of tensor for data parallel to work
    def training_step(self, batch, batch_nb):
        x, y = batch['image'], batch['seg']
        y_hat = self.forward(x)

        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        return {'loss': self.loss_func(y_hat, y),
                'progress': {'learning_rate': lr}}

    def validation_step(self, batch, batch_nb):
        x, y = batch['image'], batch['seg']
        y_hat = self.forward(x)

        return {'val_loss': self.loss_func(y_hat, y),
                'batch_iou': self.iou_metric(y_hat, y),
                'batch_any_iou': self.iou_metric(y_hat[:, -1, :, :], y[:, -1, :, :]),
                'batch_fscore': self.f_score_metric(y_hat, y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_iou = torch.stack([x['batch_iou'] for x in outputs]).mean()
        val_iou_any = torch.stack([x['batch_any_iou'] for x in outputs]).mean()
        avg_fscore = torch.stack([x['batch_fscore'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss,
                'val_iou': avg_iou,
                'avg_fscore': avg_fscore,
                'val_iou_any': val_iou_any}

    def on_batch_start(self, batch):
        if self.config.scheduler['name'] == 'flat_anneal':
            flat_iter = self.config.scheduler['flat_iterations']
            anneal_iter = self.config.scheduler['anneal_iterations']
            if flat_iter <= self.global_step < flat_iter + anneal_iter:
                self.scheduler.step()

    def configure_optimizers(self):
        if self.config.optimizer == 'adam':
            optimizer = torch.optim.Adam([{'params': self.model.decoder.parameters(), 'lr': self.config.decoder_lr},
                                          {'params': self.model.encoder.parameters(), 'lr': self.config.encoder_lr}, ],
                                         lr=self.config.lr, weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'radam':
            optimizer = RAdam([{'params': self.model.decoder.parameters(), 'lr': self.config.decoder_lr},
                               {'params': self.model.encoder.parameters(), 'lr': self.config.encoder_lr}, ],
                              lr=self.config.lr, weight_decay=self.config.weight_decay)

        if self.config.scheduler['name'] == 'flat_anneal':
            self.scheduler = CosineAnnealingLR(optimizer, self.config.scheduler['anneal_iterations'],
                                               self.config.scheduler['min_lr'])

        return optimizer

    @pl.data_loader
    def train_dataloader(self):
        use_negatives = True if self.config.negative_data_steps is not None else False
        return DataLoader(IntracranialDataset(self.config, self.train_folds, augment=self.config.augment,
                                              use_negatives=use_negatives),
                          num_workers=self.config.num_workers,
                          batch_size=self.config.batch_size,
                          shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(IntracranialDataset(self.config, self.val_folds),
                          num_workers=self.config.num_workers,
                          batch_size=self.config.batch_size)
