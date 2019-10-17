from copy import deepcopy

import torch.nn as nn
import pretrainedmodels


def get_base_model(config):
    model = config.backbone
    pretrained = config.pretrained

    _available_models = ['senet154', 'se_resnet50', 'se_resnext50']

    if config.multibranch:
        input_channels = 1
    else:
        input_channels = config.num_slices

    if model not in _available_models:
        raise ValueError('Unavailable backbone, choose one from {}'.format(_available_models))

    if model == 'senet154':
        cut_point = -3
        pretrained = 'imagenet' if pretrained else None
        model = nn.Sequential(*list(pretrainedmodels.senet154(pretrained=pretrained).children())[:cut_point])

        if input_channels != 3:
            tmp = deepcopy(model[0].conv1.weight)
            model[0].conv1 = nn.Conv2d(input_channels, 64, kernel_size=(3, 3),
                                       stride=(2, 2), padding=(1, 1), bias=False)

    if model == 'se_resnext50':
        cut_point = -2
        pretrained = 'imagenet' if pretrained else None
        model = nn.Sequential(*list(pretrainedmodels.se_resnext50_32x4d(pretrained=pretrained).children())[:cut_point])

        if input_channels != 3:
            tmp = deepcopy(model[0].conv1.weight)
            model[0].conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7),
                                       stride=(2, 2), padding=(3, 3), bias=False)

    if config.multibranch:
        model[0].conv1.weight.data.fill_(0.)
        model[0].conv1.weight[:, 0, :, :].data.copy_(tmp[:, 0, :, :])
    elif input_channels > 3 and not config.multibranch:
        diff = (input_channels - 3) // 2

        model[0].conv1.weight.data.fill_(0.)
        model[0].conv1.weight[:, diff:diff + 3, :, :].data.copy_(tmp)

    return model
