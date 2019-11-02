import itertools

import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import log_loss
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
import numpy as np

from rsna19.data.dataset_2dc import IntracranialDataset
from rsna19.models.commons.attention import ContextualAttention, SpatialAttention
from rsna19.models.commons.balancing_sampler import BalancedBatchSampler
import rsna19.models.commons.metrics as metrics
from rsna19.models.commons.radam import RAdam
from rsna19.models.commons.concat_pool import concat_pool
from rsna19.models.commons.get_base_model import get_base_model


class Classifier2DC(pl.LightningModule):
    # 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any'
    _CLASS_WEIGHTS = [1, 1, 1, 1, 1, 2]

    def __init__(self, config):
        super(Classifier2DC, self).__init__()
        self.config = config

        self.train_folds = config.train_folds
        self.val_folds = config.val_folds

        self.backbone, self.num_features_backbone = get_base_model(config)

        if self.config.multibranch3d:
            self.combine_conv = nn.Conv3d(self.num_features_backbone,
                                          config.multibranch_embedding, kernel_size=3)
            self.last_linear = nn.Linear(config.multibranch_embedding, config.n_classes)
        elif self.config.multibranch:
            self.combine_conv = nn.Conv2d(self.num_features_backbone * config.num_branches,
                                          config.multibranch_embedding, kernel_size=1)
            self.last_linear = nn.Linear(config.multibranch_embedding * 2, config.n_classes)
        else:
            self.last_linear = nn.Linear(self.num_features_backbone * 2, config.n_classes)

        if self.config.dropout > 0:
            self.dropout = nn.Dropout(self.config.dropout)
        else:
            self.dropout = None

        if self.config.multibranch:
            if self.config.contextual_attention:
                self.contextual_attention = ContextualAttention(self.num_features_backbone)
            if self.config.spatial_attention:
                self.spatial_attention = SpatialAttention(self.num_features_backbone)

        self.scheduler = None

        if self.config.freeze_backbone_iterations > 0:
            self.freeze_backbone()
            self.backbone_frozen = True
        else:
            self.backbone_frozen = False

    def freeze_backbone(self):
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        if not self.config.freeze_first_layer:
            try:
                conv1 = self.backbone[0].conv1
                bn = self.backbone[0].bn1
            except AttributeError:
                conv1 = self.backbone[0]
                bn = self.backbone[1]

            for param in itertools.chain(conv1.parameters(), bn.parameters()):
                param.requires_grad = True

    def unfreeze_backbone(self):
        self.backbone.train()
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        if self.config.multibranch:
            batch_in_size = x.shape[0]

            if self.config.multibranch_channel_indices is not None:
                x = x[:, self.config.multibranch_channel_indices, :, :]

            x = x.view(batch_in_size * self.config.num_branches, self.config.multibranch_input_channels,
                       x.shape[2], x.shape[3])
            x = self.backbone(x)
            x = x.view(batch_in_size, self.config.num_branches, self.num_features_backbone, x.shape[2], x.shape[3])

            if self.config.contextual_attention:
                x = self.contextual_attention(x)

            if self.config.spatial_attention:
                x = self.spatial_attention(x)

            if self.config.multibranch3d:
                # transform to (N, C, D, H, W), where D is num_branches
                x = x.transpose(1, 2)

                x = self.combine_conv(x)
                x = F.adaptive_avg_pool3d(x, 1)
                x = x.view(x.shape[0], -1)

            else:
                x = x.view(batch_in_size, self.config.num_branches * self.num_features_backbone, x.shape[3], x.shape[4])
                x = self.combine_conv(x)
                x = concat_pool(x)
        else:
            x = self.backbone(x)
            x = concat_pool(x)

        if self.dropout is not None:
            x = self.dropout(x)
        x = self.last_linear(x)
        return x

    # training step and validation step should return tensor or nested dicts of tensor for data parallel to work
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

        return {'val_loss': F.binary_cross_entropy_with_logits(y_hat, y, weight=class_weights),
                'y_hat_np': torch.sigmoid(y_hat),
                'y_np': y}

    def validation_end(self, outputs):
        out_dict = {}

        y_hat = np.concatenate([x['y_hat_np'].cpu().numpy() for x in outputs])
        y = np.concatenate([x['y_np'].cpu().numpy() for x in outputs])

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

        avg_loss = np.stack([x['val_loss'].cpu().numpy() for x in outputs]).mean()
        out_dict['avg_val_loss'] = avg_loss

        # implementation probably used in competition, gives slightly different results than torch
        out_dict['val_loss_sklearn'] = log_loss(y.flatten(), y_hat.flatten(),
                                                sample_weight=self._CLASS_WEIGHTS * y.shape[0])

        return out_dict

    def on_batch_start(self, batch):
        if self.backbone_frozen and self.global_step >= self.config.freeze_backbone_iterations:
            self.unfreeze_backbone()
            self.backbone_frozen = False

        if self.config.scheduler['name'] == 'flat_anneal':
            flat_iter = self.config.scheduler['flat_iterations']
            anneal_iter = self.config.scheduler['anneal_iterations']
            if flat_iter <= self.global_step < flat_iter + anneal_iter:
                self.scheduler.step()
        elif self.scheduler is not None:
            self.scheduler.step()

    def configure_optimizers(self):
        initial_lr = 1 if self.config.scheduler['name'] == 'LambdaLR' else self.config.lr

        if self.config.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=initial_lr, weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'radam':
            optimizer = RAdam(self.parameters(), lr=initial_lr, weight_decay=self.config.weight_decay)

        if self.config.scheduler['name'] == 'flat_anneal':
            self.scheduler = CosineAnnealingLR(optimizer, self.config.scheduler['anneal_iterations'],
                                               self.config.scheduler['min_lr'])
        elif self.config.scheduler['name'] == 'LambdaLR':
            iter_to_lr = self.config.scheduler['iter_to_lr']

            def f(iter):
                lr = iter_to_lr[0]
                for k, v in iter_to_lr.items():
                    if k > iter:
                        break
                    lr = v
                return lr

            self.scheduler = LambdaLR(optimizer, f)

        return optimizer

    @pl.data_loader
    def train_dataloader(self):
        if self.config.balancing:
            return DataLoader(IntracranialDataset(self.config, self.train_folds, mode='train',
                                                  augment=self.config.augment, use_cq500=self.config.use_cq500),
                              num_workers=self.config.num_workers,
                              batch_sampler=BalancedBatchSampler(self.config, self.train_folds))
        else:
            return DataLoader(IntracranialDataset(self.config, self.train_folds, mode='train',
                                                  augment=self.config.augment, use_cq500=self.config.use_cq500),
                              num_workers=self.config.num_workers,
                              batch_size=self.config.batch_size,
                              shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(IntracranialDataset(self.config, self.val_folds, mode='val'),
                          num_workers=self.config.num_workers,
                          batch_size=self.config.batch_size)
