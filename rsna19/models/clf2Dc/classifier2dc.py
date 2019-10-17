import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import log_loss
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import numpy as np

from rsna19.data.dataset_2dc import IntracranialDataset
from rsna19.models.commons.balancing_sampler import BalancedBatchSampler
import rsna19.models.commons.metrics as metrics
from rsna19.models.commons.radam import RAdam
from rsna19.models.commons.concat_pool import concat_pool
from rsna19.models.commons.get_base_model import get_base_model


class Classifier2DC(pl.LightningModule):
    _NUM_FEATURES_BACKBONE = 2048
    # 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any'
    _CLASS_WEIGHTS = [1, 1, 1, 1, 1, 2]

    def __init__(self, config):
        super(Classifier2DC, self).__init__()
        self.config = config

        self.train_folds = config.train_folds
        self.val_folds = config.val_folds

        self.backbone = get_base_model(config)
        if self.config.multibranch:
            self.combine_conv = nn.Conv2d(Classifier2DC._NUM_FEATURES_BACKBONE * config.num_slices, config.multibranch_embedding, kernel_size=1)
            self.last_linear = nn.Linear(config.multibranch_embedding * 2, config.n_classes)
        else:
            self.last_linear = nn.Linear(Classifier2DC._NUM_FEATURES_BACKBONE * 2, config.n_classes)

        if self.config.dropout > 0:
            self.dropout = nn.Dropout(self.config.dropout)
        else:
            self.dropout = None

        self.scheduler = None

    def forward(self, x):
        if self.config.multibranch:
            x = x.view(self.config.batch_size*self.config.num_slices, 1, x.shape[2], x.shape[3])
        x = self.backbone(x)
        if self.config.multibranch:
            x = x.view(self.config.batch_size, Classifier2DC._NUM_FEATURES_BACKBONE * self.config.num_slices, x.shape[2], x.shape[3])
            x = self.combine_conv(x)

        x = concat_pool(x)

        if self.dropout is not None:
            x = self.dropout(x)
        x = self.last_linear(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch['image'], batch['labels']
        y_hat = self.forward(x)
        class_weights = torch.tensor(self._CLASS_WEIGHTS, dtype=torch.float32).to(y_hat.get_device())

        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        return {'loss': F.binary_cross_entropy_with_logits(y_hat, y, weight=class_weights),
                'progress': {'learning_rate': lr}}

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
        precision = metrics.precision(y_hat, y, th, True)
        f1_score_spec = metrics.f1score_spec(y_hat, y, th, True)
        roc_auc = metrics.roc_auc(y_hat, y)

        # compute per-class loss
        class_weights = torch.tensor(self._CLASS_WEIGHTS, dtype=torch.float64)
        losses = F.binary_cross_entropy(torch.tensor(y_hat, dtype=torch.float64), torch.tensor(y, dtype=torch.float64),
                                        weight=class_weights, reduction='none')
        losses = losses.mean(dim=0)

        classes = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
        for loss, acc, f1, spec, sens, roc, prec, f1_spec, class_name in \
                zip(losses, accuracy, f1_score, specificity, sensitivity, roc_auc, precision, f1_score_spec, classes):
            out_dict['{}_loss'.format(class_name)] = loss
            out_dict['{}_acc'.format(class_name)] = acc
            out_dict['{}_f1'.format(class_name)] = f1
            out_dict['{}_spec'.format(class_name)] = spec
            out_dict['{}_sens'.format(class_name)] = sens
            out_dict['{}_roc'.format(class_name)] = roc
            out_dict['{}_prec'.format(class_name)] = prec
            out_dict['{}_f1_spec'.format(class_name)] = f1_spec

        avg_loss = np.stack([x['val_loss'] for x in outputs]).mean()
        out_dict['avg_val_loss'] = avg_loss

        # implementation probably used in competition, gives slightly different results than torch
        out_dict['val_loss_sklearn'] = log_loss(y.flatten(), y_hat.flatten(),
                                                sample_weight=self._CLASS_WEIGHTS * y.shape[0])

        return out_dict

    def on_batch_start(self, batch):
        if self.config.scheduler['name'] == 'flat_anneal':
            flat_iter = self.config.scheduler['flat_iterations']
            anneal_iter = self.config.scheduler['anneal_iterations']
            if flat_iter <= self.global_step < flat_iter + anneal_iter:
                self.scheduler.step()

    def configure_optimizers(self):
        if self.config.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'radam':
            optimizer = RAdam(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        if self.config.scheduler['name'] == 'flat_anneal':
            self.scheduler = CosineAnnealingLR(optimizer, self.config.scheduler['anneal_iterations'],
                                               self.config.scheduler['min_lr'])

        return optimizer

    @pl.data_loader
    def train_dataloader(self):
        if self.config.balancing:
            return DataLoader(IntracranialDataset(self.config, self.train_folds, augment=self.config.augment),
                              num_workers=self.config.num_workers,
                              batch_sampler=BalancedBatchSampler(self.config, self.train_folds))
        else:
            return DataLoader(IntracranialDataset(self.config, self.train_folds, augment=self.config.augment),
                              num_workers=self.config.num_workers,
                              batch_size=self.config.batch_size,
                              shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(IntracranialDataset(self.config, self.val_folds),
                          num_workers=self.config.num_workers,
                          batch_size=self.config.batch_size)
