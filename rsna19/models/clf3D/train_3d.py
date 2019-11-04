import argparse
import collections
import os

import adabound as adabound
import numpy as np
import torch
import torch.optim as optim
import torchsummary as torchsummary
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
from rsna19.data import dataset, dataset_3d_v2
import albumentations
import albumentations.pytorch
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from rsna19.configs.base_config import BaseConfig
from rsna19.models.commons import radam
from rsna19.models.clf3D.experiments_3d import MODELS
from torch.utils.tensorboard import SummaryWriter
import math

from rsna19.models.clf2D.train import build_model_str, log_metrics


class CosineAnnealingLRWithRestarts(torch.optim.lr_scheduler._LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr,
    :math:`T_{mult}` is the multiplicative factor of T_max and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}\cdot T_{mult}}\pi))
    When last_epoch=-1, sets initial lr as lr.
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        T_mult (float): Multiplicative factor of T_max. Default: 2.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, T_mult=2):
        self.T_max = T_max
        self.Ti = T_max
        self.eta_min = eta_min
        self.T_mult = T_mult
        self.cycle = 0
        super().__init__(optimizer, last_epoch)

    def step(self, epoch=None, metrics=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            if epoch == self.Ti:
                epoch = 0
                self.cycle += 1
        else:
            self.cycle = int(math.floor(math.log(epoch / self.T_max * (self.T_mult - 1) + 1, self.T_mult)))
            epoch -= sum([self.T_max * self.T_mult ** x for x in range(self.cycle)])
        self.last_epoch = epoch
        self.Ti = self.T_max * self.T_mult ** self.cycle
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.Ti)) / 2
                for base_lr in self.base_lrs]


def check_CosineAnnealingLRWithRestarts():
    conv1 = torch.nn.Conv2d(1, 1, 1)
    opt = torch.optim.SGD([{'params': conv1.parameters()}], lr=0.05)
    scheduler = CosineAnnealingLRWithRestarts(opt, T_max=8, T_mult=1.2)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=8)
    epochs = list(range(100))
    lr = []
    for e in epochs:
        scheduler.step(e)
        lr.append(scheduler.get_lr())

    plt.plot(epochs, lr)
    plt.show()


# check_CosineAnnealingLRWithRestarts()


def train(model_name, fold, run=None, resume_epoch=-1):
    model_str = build_model_str(model_name, fold, run)

    model_info = MODELS[model_name]

    checkpoints_dir = f'{BaseConfig.checkpoints_dir}/{model_str}'
    tensorboard_dir = f'{BaseConfig.tensorboard_dir}/{model_str}'
    oof_dir = f'{BaseConfig.oof_dir}/{model_str}'
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(oof_dir, exist_ok=True)
    print('\n', model_name, '\n')

    logger = SummaryWriter(log_dir=tensorboard_dir)

    model = model_info.factory(**model_info.args)
    model = model.cuda()

    # try:
    #     torchsummary.summary(model, (8, 400, 400))
    #     print('\n', model_name, '\n')
    # except:
    #     raise
    #     pass

    model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    dataset_train = dataset_3d_v2.IntracranialDataset(
        csv_file='5fold-rev3.csv',
        folds=[f for f in range(BaseConfig.nb_folds) if f != fold],
        random_slice=True,
        preprocess_func=albumentations.Compose([
            albumentations.ShiftScaleRotate(shift_limit=16./256, scale_limit=0.05, rotate_limit=30,
                                            interpolation=cv2.INTER_LINEAR,
                                            border_mode=cv2.BORDER_REPLICATE,
                                            p=0.75),
            albumentations.Flip(),
            albumentations.RandomRotate90(),
            albumentations.pytorch.ToTensorV2()
        ]),
        **model_info.dataset_args
    )

    dataset_valid = dataset_3d_v2.IntracranialDataset(
        csv_file='5fold.csv',
        folds=[fold],
        random_slice=False,
        return_all_slices=True,
        preprocess_func=albumentations.pytorch.ToTensorV2(),
        **model_info.dataset_args
    )

    model.train()
    if model_info.optimiser == 'radam':
        optimizer = radam.RAdam(model.parameters(), lr=model_info.initial_lr)
    elif model_info.optimiser == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=model_info.initial_lr, momentum=0.95, nesterov=True)
    elif model_info.optimiser == 'adabound':
        optimizer = adabound.AdaBound(model.parameters(), lr=model_info.initial_lr, final_lr=0.1)

    milestones = [32, 48, 64]
    if model_info.optimiser_milestones:
        milestones = model_info.optimiser_milestones

    if model_info.scheduler == 'steps':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)
    elif model_info.scheduler == 'cos_restarts':
        scheduler = CosineAnnealingLRWithRestarts(optimizer=optimizer, T_max=8, T_mult=1.2)

    print(f'Num training images: {len(dataset_train)} validation images: {len(dataset_valid)}')

    if resume_epoch > -1:
        checkpoint = torch.load(f'{checkpoints_dir}/{resume_epoch:03}.pt')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    data_loaders = {
        'train': DataLoader(
            dataset_train,
            num_workers=16,
            shuffle=True,
            drop_last=True,
            batch_size=model_info.batch_size),
        'val': DataLoader(
            dataset_valid,
            shuffle=False,
            num_workers=4,
            batch_size=1)
    }

    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 2.0]).cuda()

    def criterium(y_pred, y_true):
        y_pred = y_pred.reshape(-1, 6)
        y_true = y_true.reshape(-1, 6)
        cw = class_weights.repeat(y_pred.shape[0], 1)
        return F.binary_cross_entropy_with_logits(y_pred, y_true, cw)

    # fit new layers first:
    if resume_epoch == -1 and model_info.is_pretrained:
        model.train()
        model.module.freeze_encoder()
        data_loader = data_loaders['train']
        pre_fit_steps = len(dataset_train) // model_info.batch_size // 8
        data_iter = tqdm(enumerate(data_loader), total=pre_fit_steps)
        epoch_loss = []
        initial_optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
        # initial_optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
        for iter_num, data in data_iter:
            if iter_num > pre_fit_steps:
                break
            with torch.set_grad_enabled(True):
                img = data['image'].float().cuda()
                labels = data['labels'].float().cuda()
                pred = model(img)
                loss = criterium(pred, labels)
                # loss.backward()
                (loss / model_info.accumulation_steps).backward()
                if (iter_num + 1) % model_info.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
                    initial_optimizer.step()
                    initial_optimizer.zero_grad()
                epoch_loss.append(float(loss))

                data_iter.set_description(f'Loss: Running {np.mean(epoch_loss[-100:]):1.4f} Avg {np.mean(epoch_loss):1.4f}')
        del initial_optimizer
    model.module.unfreeze_encoder()

    phase_period = {
        'train': 1,
        'val': 2
    }

    for epoch_num in range(resume_epoch+1, 80):
        for phase in ['train', 'val']:
            if epoch_num % phase_period[phase] == 0:
                model.train(phase == 'train')
                epoch_loss = []
                epoch_labels = []
                epoch_predictions = []
                epoch_sample_paths = []

                if 'on_epoch' in model.module.__dir__():
                    model.module.on_epoch(epoch_num)

                data_loader = data_loaders[phase]
                data_iter = tqdm(enumerate(data_loader), total=len(data_loader))
                for iter_num, data in data_iter:
                    img = data['image'].float().cuda()
                    labels = data['labels'].float().cuda()

                    with torch.set_grad_enabled(phase == 'train'):
                        pred = model(img)
                        loss = criterium(pred, labels)

                        if phase == 'train':
                            (loss / model_info.accumulation_steps).backward()
                            if (iter_num + 1) % model_info.accumulation_steps == 0:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), 32.0)
                                optimizer.step()
                                optimizer.zero_grad()

                        epoch_loss.append(float(loss))

                        epoch_labels.append(np.row_stack(labels.detach().cpu().numpy()))
                        epoch_predictions.append(np.row_stack(torch.sigmoid(pred).detach().cpu().numpy()))

                        # print(labels.shape, epoch_labels[-1].shape, pred.shape, epoch_predictions[-1].shape)
                        epoch_sample_paths += data['path']

                    data_iter.set_description(
                        f'{epoch_num} Loss: Running {np.mean(epoch_loss[-100:]):1.4f} Avg {np.mean(epoch_loss):1.4f}')

                epoch_labels = np.row_stack(epoch_labels)
                epoch_predictions = np.row_stack(epoch_predictions)
                if phase == 'val':
                    # recalculate loss as depth dimension is variable
                    epoch_loss_mean = float(F.binary_cross_entropy(
                        torch.from_numpy(epoch_predictions).cuda(),
                        torch.from_numpy(epoch_labels).cuda(),
                        class_weights.repeat(epoch_labels.shape[0], 1)
                    ))
                    print(epoch_loss_mean)
                    logger.add_scalar(f'loss_{phase}', epoch_loss_mean, epoch_num)
                else:
                    logger.add_scalar(f'loss_{phase}', np.mean(epoch_loss), epoch_num)
                logger.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch_num)  # scheduler.get_lr()[0]
                try:
                    log_metrics(logger=logger, phase=phase, epoch_num=epoch_num, y=epoch_labels, y_hat=epoch_predictions)
                except Exception:
                    pass

                if phase == 'val':
                    torch.save(
                        {
                            'epoch': epoch_num,
                            'sample_paths': epoch_sample_paths,
                            'epoch_labels': epoch_labels,
                            'epoch_predictions': epoch_predictions,
                        },
                        f'{oof_dir}/{epoch_num:03}.pt'
                    )

            logger.flush()

            if phase == 'val':
                scheduler.step(epoch=epoch_num)
            else:
                # print(f'{checkpoints_dir}/{epoch_num:03}.pt')
                torch.save(
                    {
                        'epoch': epoch_num,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    },
                    f'{checkpoints_dir}/{epoch_num:03}.pt'
                )


def check_score(model_name, fold, epoch, run=None):
    import sklearn.metrics

    model_str = build_model_str(model_name, fold, run)
    model_info = MODELS[model_name]

    oof_dir = f'{BaseConfig.oof_dir}/{model_str}'
    print('\n', model_str, '\n')

    pred = torch.load(f'{oof_dir}/{epoch:03}.pt')
    epoch_labels = pred['epoch_labels']
    epoch_predictions = pred['epoch_predictions']

    def double_any(d):
        return np.column_stack([d, d[:, -1:]])

    print(sklearn.metrics.log_loss(double_any(epoch_labels).flatten(), double_any(epoch_predictions).flatten()))

    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 2.0])
    # return F.binary_cross_entropy_with_logits(y_pred, y_true, class_weights.repeat(y_pred.shape[0], 1))
    print(F.binary_cross_entropy(torch.from_numpy((double_any(epoch_predictions)).flatten()),
                                 torch.from_numpy((double_any(epoch_labels)).flatten())))

    print(F.binary_cross_entropy(torch.from_numpy(epoch_predictions).reshape(-1),
                                 torch.from_numpy(epoch_labels).reshape(-1),
                                 class_weights.repeat(epoch_predictions.shape[0], 1).reshape(-1)
                                 ))

    print(F.binary_cross_entropy(torch.from_numpy(epoch_predictions),
                                 torch.from_numpy(epoch_labels),
                                 class_weights.repeat(epoch_predictions.shape[0], 1)
                                 ))

    loss = F.binary_cross_entropy(
        torch.from_numpy(epoch_predictions),
        torch.from_numpy(epoch_labels),
        class_weights.repeat(epoch_predictions.shape[0], 1),
        reduction='none')

    print(loss.shape)
    loss = loss.cpu().detach().numpy()
    loss = np.mean(loss, axis=1)

    # plt.hist(loss, bins=1024)
    plt.plot(np.sort(-1*loss)*-1)
    plt.axvline()
    plt.axhline()
    plt.show()

    # clamp_values = np.arange(-8, -2.2, 0.1)
    # loss = [sklearn.metrics.log_loss(double_any(epoch_labels).flatten(), double_any(epoch_predictions).flatten(), eps=18**c) for c in clamp_values]
    # plt.plot(clamp_values, loss)
    # print(min(loss))
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, default='check')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--run', type=str, default='')
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--epoch', type=int, default=-1)

    parser.add_argument('--resume_weights', type=str, default='')
    parser.add_argument('--resume_epoch', type=int, default=-1)

    args = parser.parse_args()
    action = args.action

    if action == 'train':
        try:
            train(model_name=args.model, run=args.run, fold=args.fold, resume_epoch=args.resume_epoch)
        except KeyboardInterrupt:
            pass

    if action == 'check_score':
        check_score(model_name=args.model, run=args.run, fold=args.fold, epoch=args.epoch)
