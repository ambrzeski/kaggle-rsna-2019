import argparse
import collections
import os

import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
from rsna19.data import dataset
import albumentations
import albumentations.pytorch
import cv2
import matplotlib.pyplot as plt
import sklearn.metrics
from apex import amp

import torch.nn as nn
import torch.nn.functional as F
from rsna19.configs.base_config import BaseConfig
from rsna19.models.commons import radam
from rsna19.models.commons import metrics
from rsna19.models.clf2D.experiments import MODELS
from torch.utils.tensorboard import SummaryWriter


def build_model_str(model_name, fold, run):
    run_str = '' if not run else f'_{run}'
    fold_str = '' if fold == -1 else f'_fold_{fold}'

    return f'{model_name}{run_str}{fold_str}'


def log_metrics(logger, phase, epoch_num, y_hat, y):
    th = 0.5
    accuracy = metrics.accuracy(y_hat, y, th, True)
    f1_score = metrics.f1score(y_hat, y, th, True)
    specificity = metrics.specificity(y_hat, y, th, True)
    sensitivity = metrics.sensitivity(y_hat, y, th, True)
    roc_auc = metrics.roc_auc(y_hat, y)

    classes = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
    for acc, f1, spec, sens, roc, class_name in zip(accuracy, f1_score, specificity, sensitivity, roc_auc, classes):
        logger.add_scalar(f'{phase}_acc_{class_name}', acc, epoch_num)
        logger.add_scalar(f'{phase}_f1_{class_name}', f1, epoch_num)
        logger.add_scalar(f'{phase}_spec_{class_name}', spec, epoch_num)
        logger.add_scalar(f'{phase}_sens_{class_name}', sens, epoch_num)
        logger.add_scalar(f'{phase}_roc_{class_name}', roc, epoch_num)

    for i, class_name in enumerate(classes):
        logger.add_scalar(f'{phase}_bce_{class_name}', sklearn.metrics.log_loss(y[:, i], y_hat[:, i]), epoch_num)


def train(model_name, fold, run=None, resume_epoch=-1, use_apex=False):
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
    #     torchsummary.summary(model, (4, 512, 512))
    #     print('\n', model_name, '\n')
    # except:
    #     raise
    #     pass

    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    augmentations = [
        albumentations.ShiftScaleRotate(shift_limit=16. / 256, scale_limit=0.05, rotate_limit=30,
                                        interpolation=cv2.INTER_LINEAR,
                                        border_mode=cv2.BORDER_REPLICATE,
                                        p=0.80),
    ]
    if model_info.use_vflip:
        augmentations += [
            albumentations.Flip(),
            albumentations.RandomRotate90()
        ]
    else:
        augmentations += [
            albumentations.HorizontalFlip()
        ]

    dataset_train = dataset.IntracranialDataset(
        csv_file='5fold-test-rev3.csv',
        folds=[f for f in range(BaseConfig.nb_folds) if f != fold],
        preprocess_func=albumentations.Compose(augmentations),
        **model_info.dataset_args
    )

    dataset_valid = dataset.IntracranialDataset(
        csv_file='5fold-test-rev3.csv',
        folds=[fold],
        preprocess_func=None,
        **model_info.dataset_args
    )

    data_loaders = {
        'train': DataLoader(dataset_train,
                            num_workers=8,
                            shuffle=True,
                            batch_size=model_info.batch_size),
        'val':   DataLoader(dataset_valid,
                            shuffle=False,
                            num_workers=8,
                            batch_size=model_info.batch_size)
    }

    if model_info.single_slice_steps > 0:
        augmentations = [
                albumentations.ShiftScaleRotate(shift_limit=16. / 256, scale_limit=0.05, rotate_limit=30,
                                                interpolation=cv2.INTER_LINEAR,
                                                border_mode=cv2.BORDER_REPLICATE,
                                                p=0.80),
            ]
        if model_info.use_vflip:
            augmentations += [
                albumentations.Flip(),
                albumentations.RandomRotate90()
            ]
        else:
            augmentations += [
                albumentations.HorizontalFlip()
            ]

        dataset_train_1_slice = dataset.IntracranialDataset(
            csv_file='5fold-test-rev3.csv',
            folds=[f for f in range(BaseConfig.nb_folds) if f != fold],
            preprocess_func=albumentations.Compose(augmentations),
            **{**model_info.dataset_args, "num_slices": 1}
        )

        dataset_valid_1_slice = dataset.IntracranialDataset(
            csv_file='5fold-test-rev3.csv',
            folds=[fold],
            preprocess_func=None,
            **{**model_info.dataset_args, "num_slices": 1}
        )

        data_loaders['train_1_slice'] = DataLoader(
            dataset_train_1_slice,
            num_workers=8,
            shuffle=True,
            batch_size=model_info.batch_size*2)
        data_loaders['val_1_slice'] = DataLoader(
            dataset_valid_1_slice,
            shuffle=False,
            num_workers=8,
            batch_size=model_info.batch_size*2)

    model.train()

    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 2.0]).cuda()

    def criterium(y_pred,y_true):
        return F.binary_cross_entropy_with_logits(y_pred, y_true, class_weights.repeat(y_pred.shape[0], 1))

    # fit the new layers first:
    if resume_epoch == -1 and model_info.is_pretrained:
        model.train()
        model.freeze_encoder()
        data_loader = data_loaders.get('train_1_slice', data_loaders['train'])
        pre_fit_steps = 40000 // model_info.batch_size
        data_iter = tqdm(enumerate(data_loader), total=pre_fit_steps)
        epoch_loss = []
        initial_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for iter_num, data in data_iter:
            if iter_num > pre_fit_steps:
                break
            with torch.set_grad_enabled(True):
                img = data['image'].float().cuda()
                labels = data['labels'].cuda()
                pred = model(img)
                loss = criterium(pred, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
                initial_optimizer.step()
                initial_optimizer.zero_grad()
                epoch_loss.append(float(loss))

                data_iter.set_description(f'Loss: Running {np.mean(epoch_loss[-500:]):1.4f} Avg {np.mean(epoch_loss):1.4f}')
        model.unfreeze_encoder()

    optimizer = radam.RAdam(model.parameters(), lr=model_info.initial_lr)
    if use_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    milestones = [5, 10, 16]
    if model_info.optimiser_milestones:
        milestones = model_info.optimiser_milestones
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)

    print(f'Num training images: {len(dataset_train)} validation images: {len(dataset_valid)}')

    if resume_epoch > -1:
        checkpoint = torch.load(f'{checkpoints_dir}/{resume_epoch:03}.pt')
        print('load', f'{checkpoints_dir}/{resume_epoch:03}.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'amp' in checkpoint:
            amp.load_state_dict(checkpoint['amp'])

    for epoch_num in range(resume_epoch+1, 7):
        for phase in ['train', 'val']:
            model.train(phase == 'train')
            epoch_loss = []
            epoch_labels = []
            epoch_predictions = []
            epoch_sample_paths = []

            if 'on_epoch' in model.__dir__():
                model.on_epoch(epoch_num)

            if epoch_num < model_info.single_slice_steps:
                data_loader = data_loaders[phase+'_1_slice']
                print("use 1 slice input")
            else:
                data_loader = data_loaders[phase]
                print("use N slices input")

            # if epoch_num == model_info.single_slice_steps:
            #     print("train only conv slices/fn layers")
            #     model.module.freeze_encoder_full()
            #
            # if epoch_num == model_info.single_slice_steps+1:
            #     print("train all")
            #     model.module.unfreeze_encoder()
            #
            # if -1 < model_info.freeze_bn_step <= epoch_num:
            #     print("freeze bn")
            #     model.module.freeze_bn()

            data_iter = tqdm(enumerate(data_loader), total=len(data_loader), ncols=200)
            for iter_num, data in data_iter:
                img = data['image'].float().cuda()
                labels = data['labels'].float().cuda()

                with torch.set_grad_enabled(phase == 'train'):
                    # if epoch_num == model_info.single_slice_steps and phase == 'train':
                    #     with torch.set_grad_enabled(False):
                    #         model_x = model(img, output_before_combine_slices=True)
                    #     with torch.set_grad_enabled(True):
                    #         pred = model(model_x.detach(), train_last_layers_only=True)
                    # else:
                    pred = model(img)
                    loss = criterium(pred, labels)

                    if phase == 'train':
                        if use_apex:
                            with amp.scale_loss(loss / model_info.accumulation_steps, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            (loss / model_info.accumulation_steps).backward()

                        if (iter_num + 1) % model_info.accumulation_steps == 0:
                            # if not use_apex:
                            #     torch.nn.utils.clip_grad_norm_(model.parameters(), 32.0)
                            optimizer.step()
                            optimizer.zero_grad()

                    epoch_loss.append(float(loss))

                    epoch_labels.append(labels.detach().cpu().numpy())
                    epoch_predictions.append(torch.sigmoid(pred).detach().cpu().numpy())
                    epoch_sample_paths += data['path']

                data_iter.set_description(
                    f'{epoch_num} Loss: Running {np.mean(epoch_loss[-1000:]):1.4f} Avg {np.mean(epoch_loss):1.4f}')

            logger.add_scalar(f'loss_{phase}', np.mean(epoch_loss), epoch_num)
            logger.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch_num)  # scheduler.get_lr()[0]
            try:
                epoch_labels = np.row_stack(epoch_labels)
                epoch_predictions = np.row_stack(epoch_predictions)
                print(epoch_labels.shape, epoch_predictions.shape)
                log_metrics(logger=logger, phase=phase, epoch_num=epoch_num, y=epoch_labels, y_hat=epoch_predictions)
            except Exception:
                pass
            logger.flush()

            if phase == 'val':
                scheduler.step(epoch=epoch_num)
                torch.save(
                    {
                        'epoch': epoch_num,
                        'sample_paths': epoch_sample_paths,
                        'epoch_labels': epoch_labels,
                        'epoch_predictions': epoch_predictions,
                    },
                    f'{oof_dir}/{epoch_num:03}.pt'
                )
            else:
                # print(f'{checkpoints_dir}/{epoch_num:03}.pt')
                torch.save(
                    {
                        'epoch': epoch_num,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'amp': amp.state_dict()
                    },
                    f'{checkpoints_dir}/{epoch_num:03}.pt'
                )


def check_heatmap(model_name, fold, epoch, run=None):
    model_str = build_model_str(model_name, fold, run)
    model_info = MODELS[model_name]

    checkpoints_dir = f'{BaseConfig.checkpoints_dir}/{model_str}'
    print('\n', model_name, '\n')

    model = model_info.factory(**model_info.args)
    model = model.cpu()

    dataset_valid = dataset.IntracranialDataset(
        csv_file='5fold.csv',
        folds=[fold],
        preprocess_func=albumentations.pytorch.ToTensorV2(),
        **model_info.dataset_args
    )

    model.eval()
    checkpoint = torch.load(f'{checkpoints_dir}/{epoch:03}.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cpu()

    batch_size = 1

    data_loader = DataLoader(dataset_valid,
                          shuffle=False,
                          num_workers=16,
                          batch_size=batch_size)

    data_iter = tqdm(enumerate(data_loader), total=len(data_loader))
    for iter_num, data in data_iter:
        img = data['image'].float().cpu()
        labels = data['labels'].detach().numpy()

        with torch.set_grad_enabled(False):
            pred2d, heatmap, pred = model(img, output_heatmap=True, output_per_pixel=True)
            heatmap *= np.prod(heatmap.shape[1:])

            pred2d = (pred2d[0]).detach().cpu().numpy() * 0.1

            fig, ax = plt.subplots(2, 4)

            for i in range(batch_size):
                print(labels[i], torch.sigmoid(pred[i]))
                ax[0, 0].imshow(img[i, 0].cpu().detach().numpy(), cmap='gray')
                ax[0, 1].imshow(heatmap[i, 0].cpu().detach().numpy(), cmap='gray')
                ax[0, 2].imshow(pred2d[0], cmap='gray', vmin=0, vmax=1)
                ax[0, 3].imshow(pred2d[1], cmap='gray', vmin=0, vmax=1)
                ax[1, 0].imshow(pred2d[2], cmap='gray', vmin=0, vmax=1)
                ax[1, 1].imshow(pred2d[3], cmap='gray', vmin=0, vmax=1)
                ax[1, 2].imshow(pred2d[4], cmap='gray', vmin=0, vmax=1)
                ax[1, 3].imshow(pred2d[5], cmap='gray', vmin=0, vmax=1)

            plt.show()


def check_windows(model_name, fold, epoch, run=None):
    model_str = build_model_str(model_name, fold, run)
    model_info = MODELS[model_name]

    checkpoints_dir = f'{BaseConfig.checkpoints_dir}/{model_str}'
    print('\n', model_name, '\n')

    model = model_info.factory(**model_info.args)
    model = model.cpu()

    dataset_valid = dataset.IntracranialDataset(
        csv_file='5fold.csv',
        folds=[fold],
        preprocess_func=albumentations.pytorch.ToTensorV2(),
        **model_info.dataset_args
    )

    model.eval()
    checkpoint = torch.load(f'{checkpoints_dir}/{epoch:03}.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cpu()

    w = model.windows_conv.weight.detach().cpu().numpy().flatten()
    b = model.windows_conv.bias.detach().cpu().numpy()
    print(w, b)
    for wi, bi in zip(w, b):
        print(f'{-int(bi/wi*1000)} +- {int(abs(1000/wi))}')

    batch_size = 1

    data_loader = DataLoader(dataset_valid,
                          shuffle=False,
                          num_workers=16,
                          batch_size=batch_size)

    data_iter = tqdm(enumerate(data_loader), total=len(data_loader))
    for iter_num, data in data_iter:
        img = data['image'].float().cpu()
        labels = data['labels'].detach().numpy()

        with torch.set_grad_enabled(False):
            windowed_img = model.windows_conv(img)
            windowed_img = F.relu6(windowed_img).cpu().numpy()

            fig, ax = plt.subplots(4, 4)

            for batch in range(batch_size):
                print(labels[batch], data['path'][batch])
                for j in range(4):
                    for k in range(4):
                        ax[j, k].imshow(windowed_img[batch, j*4+k], cmap='gray')

            plt.show()


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
    parser.add_argument('--apex', action='store_true')

    parser.add_argument('--resume_weights', type=str, default='')
    parser.add_argument('--resume_epoch', type=int, default=-1)

    args = parser.parse_args()
    action = args.action

    if action == 'train':
        try:
            train(model_name=args.model, run=args.run, fold=args.fold, resume_epoch=args.resume_epoch, use_apex=args.apex)
        except KeyboardInterrupt:
            pass

    if action == 'check_heatmap':
        check_heatmap(model_name=args.model, run=args.run, fold=args.fold, epoch=args.epoch)

    if action == 'check_windows':
        check_windows(model_name=args.model, run=args.run, fold=args.fold, epoch=args.epoch)

    if action == 'check_score':
        check_score(model_name=args.model, run=args.run, fold=args.fold, epoch=args.epoch)
