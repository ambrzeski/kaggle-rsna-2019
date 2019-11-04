import torch
from copy import deepcopy

import torch.nn as nn
import pretrainedmodels


def get_base_model(config):
    model_name = config.backbone
    pretrained = config.pretrained

    if pretrained is not None and pretrained != 'imagenet':
        weights_path = pretrained
        pretrained = None
    else:
        weights_path = None

    if config.multibranch:
        input_channels = config.multibranch_input_channels
    else:
        input_channels = config.num_slices
        if hasattr(config, 'append_masks') and config.append_masks:
            input_channels *= 2

    _available_models = ['senet154', 'se_resnext50', 'resnet34', 'resnet18']

    if model_name == 'senet154':
        cut_point = -3
        model = nn.Sequential(*list(pretrainedmodels.senet154(pretrained=pretrained).children())[:cut_point])
        num_features = 2048
    elif model_name == 'se_resnext50':
        cut_point = -2
        model = nn.Sequential(*list(pretrainedmodels.se_resnext50_32x4d(pretrained=pretrained).children())[:cut_point])
        num_features = 2048
    elif model_name == 'resnet34':
        cut_point = -2
        model = nn.Sequential(*list(pretrainedmodels.resnet34(pretrained=pretrained).children())[:cut_point])
        num_features = 512
    elif model_name == 'resnet18':
        cut_point = -2
        model = nn.Sequential(*list(pretrainedmodels.resnet18(pretrained=pretrained).children())[:cut_point])
        num_features = 512
    else:
        raise ValueError('Unavailable backbone, choose one from {}'.format(_available_models))

    if model_name in ['senet154', 'se_resnext50']:
        conv1 = model[0].conv1
    else:
        conv1 = model[0]

    if input_channels != 3:
        conv1_weights = deepcopy(conv1.weight)
        new_conv1 = nn.Conv2d(input_channels, conv1.out_channels, kernel_size=conv1.kernel_size,
                              stride=conv1.stride, padding=conv1.padding, bias=conv1.bias)

        if weights_path is None:
            if input_channels == 1:
                new_conv1.weight.data.fill_(0.)
                new_conv1.weight[:, 0, :, :].data.copy_(conv1_weights[:, 0, :, :])
            elif input_channels > 3:
                diff = (input_channels - 3) // 2

                new_conv1.weight.data.fill_(0.)
                new_conv1.weight[:, diff:diff + 3, :, :].data.copy_(conv1_weights)

        if model_name in ['senet154', 'se_resnext50']:
            model[0].conv1 = new_conv1
        else:
            model[0] = new_conv1

    if weights_path is not None:
        if model_name in ['senet154', 'se_resnext50']:
            conv1_str = '0.conv1.weight'
        else:
            conv1_str = '0.weight'
        weights = load_base_weights(weights_path, input_channels, conv1_str)
        model.load_state_dict(weights)

    return model, num_features


def load_base_weights(weights_path, input_channels, conv1_str):
    weights = torch.load(weights_path, map_location='cpu')['state_dict']
    weights = {k.replace('backbone.', ''): v for k, v in weights.items() if not k.startswith('last')}

    conv1_weights = weights[conv1_str]
    new_shape = list(conv1_weights.shape)
    new_shape[1] = input_channels
    new_conv1_weights = torch.zeros(new_shape)

    mid_c = conv1_weights.shape[1] // 2
    new_mid_c = new_conv1_weights.shape[1] // 2
    copied_channels = min(conv1_weights.shape[1], new_conv1_weights.shape[1])

    new_conv1_weights[:, new_mid_c - copied_channels // 2: new_mid_c + copied_channels // 2 + 1, :, :] = \
        conv1_weights[:, mid_c - copied_channels // 2: mid_c + copied_channels // 2 + 1, :, :]

    weights[conv1_str] = new_conv1_weights

    return weights
