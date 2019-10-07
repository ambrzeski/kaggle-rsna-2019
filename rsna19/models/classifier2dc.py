
import pretrainedmodels
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from rsna19.data.dataset import IntracranialDataset


class Classifier2DC(pl.LightningModule):
    _NUM_FEATURES_BACKBONE = 2048

    @staticmethod
    def get_base_model(model, pretrained):
        _available_models = ['senet154', 'se_resnet50', 'se_resnext50']

        if model not in _available_models:
            raise ValueError('Unavailable backbone, choose one from {}'.format(_available_models))

        if model == 'senet154':
            cut_point = -3
            pretrained = 'imagenet' if pretrained else None
            return nn.Sequential(*list(pretrainedmodels.senet154(pretrained=pretrained).children())[:cut_point])

        if model == 'se_resnext50':
            cut_point = -2
            pretrained = 'imagenet' if pretrained else None
            return nn.Sequential(*list(pretrainedmodels.se_resnext50_32x4d(pretrained=pretrained).children())[:cut_point])

        if model == 'se_resnet50':
            cut_point = -2
            pretrained = 'imagenet' if pretrained else None
            return nn.Sequential(*list(pretrainedmodels.se_resnet50(pretrained=pretrained).children())[:cut_point])

    def __init__(self, config):
        super(Classifier2DC, self).__init__()

        self.config = config

        self.train_folds = config.train_folds
        self.val_folds = config.val_folds

        self.backbone = self.get_base_model(config.backbone, config.pretrained)
        self.last_linear = nn.Linear(Classifier2DC._NUM_FEATURES_BACKBONE, config.n_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch['image'], batch['labels']
        y_hat = self.forward(x)
        return {'loss': F.binary_cross_entropy_with_logits(y_hat, y)}

    def validation_step(self, batch, batch_nb):
        x, y = batch['image'], batch['labels']
        y_hat = self.forward(x)
        return {'val_loss': F.binary_cross_entropy_with_logits(y_hat, y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(IntracranialDataset(self.config, self.train_folds),
                          shuffle=True,
                          num_workers=self.config.num_workers,
                          batch_size=self.config.batch_size)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(IntracranialDataset(self.config, self.val_folds),
                          num_workers=self.config.num_workers,
                          batch_size=self.config.batch_size)

