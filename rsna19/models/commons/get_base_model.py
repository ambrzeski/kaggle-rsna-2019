import torch
from copy import deepcopy

import torch.nn as nn
import pretrainedmodels


def get_base_model(config):
    model = config.backbone
    pretrained = config.pretrained

    if pretrained is not None and pretrained != 'imagenet':
        weights_path = pretrained
        pretrained = None
    else:
        weights_path = None

    _available_models = ['senet154', 'se_resnet50', 'se_resnext50']

    if config.multibranch:
        input_channels = 1
    else:
        input_channels = config.num_slices

    if model not in _available_models:
        raise ValueError('Unavailable backbone, choose one from {}'.format(_available_models))

    if model == 'senet154':
        cut_point = -3
        model = nn.Sequential(*list(pretrainedmodels.senet154(pretrained=pretrained).children())[:cut_point])

        if input_channels != 3:
            conv1_weights = deepcopy(model[0].conv1.weight)
            model[0].conv1 = nn.Conv2d(input_channels, 64, kernel_size=(3, 3),
                                       stride=(2, 2), padding=(1, 1), bias=False)

    if model == 'se_resnext50':
        cut_point = -2
        model = nn.Sequential(*list(pretrainedmodels.se_resnext50_32x4d(pretrained=pretrained).children())[:cut_point])

        if input_channels != 3:
            conv1_weights = deepcopy(model[0].conv1.weight)
            model[0].conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7),
                                       stride=(2, 2), padding=(3, 3), bias=False)

    if weights_path is None:
        if input_channels == 1:
            model[0].conv1.weight.data.fill_(0.)
            model[0].conv1.weight[:, 0, :, :].data.copy_(conv1_weights[:, 0, :, :])
        elif input_channels > 3 and not config.multibranch:
            diff = (input_channels - 3) // 2

            model[0].conv1.weight.data.fill_(0.)
            model[0].conv1.weight[:, diff:diff + 3, :, :].data.copy_(conv1_weights)

    else:
        weights = load_base_weights(weights_path, input_channels)
        model.load_state_dict(weights)

    return model


def load_base_weights(weights_path, input_channels):
    weights = torch.load(weights_path, map_location='cpu')['state_dict']
    weights = {k.replace('backbone.', ''): v for k, v in weights.items() if not k.startswith('last')}

    conv1_weights = weights['0.conv1.weight']
    new_shape = list(conv1_weights.shape)
    new_shape[1] = input_channels
    new_conv1_weights = torch.zeros(new_shape)

    mid_c = conv1_weights.shape[1] // 2
    new_mid_c = new_conv1_weights.shape[1] // 2
    copied_channels = min(conv1_weights.shape[1], new_conv1_weights.shape[1])

    new_conv1_weights[:, new_mid_c - copied_channels // 2: new_mid_c + copied_channels // 2 + 1, :, :] = \
        conv1_weights[:, mid_c - copied_channels // 2: mid_c + copied_channels // 2 + 1, :, :]

    weights['0.conv1.weight'] = new_conv1_weights

    return weights
