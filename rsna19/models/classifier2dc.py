
import pretrainedmodels
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np

from rsna19.data.dataset import IntracranialDataset
from rsna19.models.balancing_sampler import BalancedBatchSampler
import rsna19.models.metrics as metrics


class Classifier2DC(pl.LightningModule):
    _NUM_FEATURES_BACKBONE = 2048
    # 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any'
    _CLASS_WEIGHTS = [1, 1, 1, 1, 1, 2]

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
        class_weights = torch.tensor(self._CLASS_WEIGHTS, dtype=torch.float32).to(y_hat.get_device())
        return {'loss': F.binary_cross_entropy_with_logits(y_hat, y, weight=class_weights)}

    def validation_step(self, batch, batch_nb):
        x, y = batch['image'], batch['labels']
        y_hat = self.forward(x)
        class_weights = torch.tensor(self._CLASS_WEIGHTS, dtype=torch.float32).to(y_hat.get_device())
        return {'val_loss': F.binary_cross_entropy_with_logits(y_hat, y, weight=class_weights).cpu().numpy(),
                'y_hat_np': torch.sigmoid(y_hat).cpu().numpy(),
                'y_np': y.cpu().numpy()}

    def validation_end(self, outputs):
        out_dict = {}

        y_hat = np.concatenate([x['y_hat_np'] for x in outputs])
        y = np.concatenate([x['y_np'] for x in outputs])

        th = 0.5
        accuracy = metrics.accuracy(y_hat, y, th, True)
        f1_score = metrics.f1score(y_hat, y, th, True)
        specificity = metrics.specificity(y_hat, y, th, True)
        sensitivity = metrics.sensitivity(y_hat, y, th, True)
        precision = metrics.precision(y_hat, y)
        f1__score_spec = metrics.f1score_spec(y_hat, y)
        roc_auc = metrics.roc_auc(y_hat, y)

        classes = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
        for acc, f1, spec, sens, roc, prec, f1_spec, class_name in \
                zip(accuracy, f1_score, specificity, sensitivity, roc_auc, precision, f1__score_spec, classes):
            out_dict['{}_acc'.format(class_name)] = acc
            out_dict['{}_f1'.format(class_name)] = f1
            out_dict['{}_spec'.format(class_name)] = spec
            out_dict['{}_sens'.format(class_name)] = sens
            out_dict['{}_roc'.format(class_name)] = roc
            out_dict['{}_prec'.format(class_name)] = prec
            out_dict['{}_f1_spec'.format(class_name)] = f1_spec

        avg_loss = np.stack([x['val_loss'] for x in outputs]).mean()
        out_dict['avg_val_loss'] = avg_loss

        return out_dict

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(IntracranialDataset(self.config, self.train_folds),
                          num_workers=self.config.num_workers,
                          batch_sampler=BalancedBatchSampler(self.config, self.train_folds))

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(IntracranialDataset(self.config, self.val_folds),
                          num_workers=self.config.num_workers,
                          batch_size=self.config.batch_size)

