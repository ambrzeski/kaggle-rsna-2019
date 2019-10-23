import torch
from torch import nn
from rsna19.models.clf3D import resnet


def generate_model(config):
    resnet_class = getattr(resnet, config.backbone)
    model = resnet_class(
        shortcut_type=config.resnet_shortcut,
        dropout=config.dropout
    )

    model = model.cuda() 
    net_dict = model.state_dict()

    if config.pretrained:
        print('Loading pretrained model {}'.format(config.pretrained))
        pretrain = torch.load(config.pretrained)
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
         
        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

    return model
