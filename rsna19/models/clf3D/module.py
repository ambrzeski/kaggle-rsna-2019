import pretrainedmodels
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import log_loss
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy

from rsna19.data.dataset_3d import IntracranialDataset3D
from rsna19.models.commons.balancing_sampler import BalancedBatchSampler
import rsna19.models.commons.metrics as metrics
from rsna19.models.commons.radam import RAdam
from rsna19.models.clf3D.utils import generate_model


class MedicalNetModule(pl.LightningModule):
    # 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any'
    _CLASS_WEIGHTS = [1, 1, 1, 1, 1, 2]

    def __init__(self, config):
        super(MedicalNetModule, self).__init__()

        self.config = config

        self.train_folds = config.train_folds
        self.val_folds = config.val_folds

        self.backbone = generate_model(config)

    def forward(self, x):
        y = self.backbone(x)
        return y

    def training_step(self, batch, batch_nb):
        x, y = batch['image'], batch['labels']
        y_hat = self.forward(x)
        class_weights = torch.tensor(self._CLASS_WEIGHTS, dtype=torch.float32).to(y_hat.get_device())
        return {'loss': F.binary_cross_entropy_with_logits(y_hat, y, weight=class_weights)}

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

    def configure_optimizers(self):
        new_parameters = []
        print("Layers with boosted learning rate:")
        for pname, p in self.named_parameters():
            for layer_name in self.config.new_layer_names:
                if layer_name in pname:
                    print(pname)
                    new_parameters.append(p)
                    break

        new_parameters_id = list(map(id, new_parameters))
        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, self.parameters()))
        parameters = [
            {'params': base_parameters, 'lr': self.config.lr},
            {'params': new_parameters, 'lr': self.config.lr * self.config.new_params_lr_boost}
        ]

        if self.config.optimizer == 'adam':
            return torch.optim.Adam(parameters, lr=self.config.lr, weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'radam':
            return RAdam(parameters, lr=self.config.lr, weight_decay=self.config.weight_decay)

    @pl.data_loader
    def train_dataloader(self):
        if self.config.balancing:
            return DataLoader(IntracranialDataset3D(self.config, self.train_folds, augment=self.config.augment),
                              num_workers=self.config.num_workers,
                              batch_sampler=BalancedBatchSampler(self.config, self.train_folds))
        else:
            return DataLoader(IntracranialDataset3D(self.config, self.train_folds, augment=self.config.augment),
                              num_workers=self.config.num_workers,
                              batch_size=self.config.batch_size,
                              shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(IntracranialDataset3D(self.config, self.val_folds),
                          num_workers=self.config.num_workers,
                          batch_size=self.config.batch_size)
