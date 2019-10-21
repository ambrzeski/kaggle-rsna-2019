import torch
import torch.nn as nn
import torch.nn.functional as F

import pretrainedmodels


class ClassificationModelResnetCombineLast(nn.Module):
    def __init__(self,
                 base_model,
                 base_model_features,
                 base_model_l1_outputs,
                 nb_features,
                 nb_input_slices=5,
                 dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.base_model = base_model
        self.nb_features = nb_features
        self.nb_input_slices = nb_input_slices
        self.base_model_features = base_model_features

        self.l1 = nn.Conv2d(1, base_model_l1_outputs, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_model_l1_outputs)
        self.combine_conv = nn.Conv2d(base_model_features * nb_input_slices, 256, kernel_size=1)
        self.fc = nn.Linear(256*2, nb_features)

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        self.base_model.requires_grad = True
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs, output_per_pixel=False):
        res = []
        batch_size = inputs.shape[0]
        nb_input_slices = self.nb_input_slices

        x = inputs.view(batch_size*nb_input_slices, 1, inputs.shape[2], inputs.shape[3])
        x = self.l1(x)
        x = self.bn1(x)
        # TODO: batch norm here may still help when cdf used
        x = torch.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)

        base_model_features = x.shape[1]
        x = x.view(batch_size, base_model_features*nb_input_slices, x.shape[2], x.shape[3])
        x = self.combine_conv(x)

        # TODO: seems to work worse with relu here, need more testing
        # x = torch.relu(x)

        if output_per_pixel:
            res.append(F.conv2d(torch.cat([x, x], dim=1), self.fc.weight[:, :, None, None], self.fc.bias))

        avg_pool = F.avg_pool2d(x, x.shape[2:])
        max_pool = F.max_pool2d(x, x.shape[2:])
        avg_max_pool = torch.cat((avg_pool, max_pool), 1)
        x = avg_max_pool.view(avg_max_pool.size(0), -1)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, self.training)

        out = self.fc(x)

        if res:
            res.append(out)
            return res
        else:
            return out


class ClassificationModelResnetCombineFirst(nn.Module):
    def __init__(self,
                 base_model,
                 base_model_features,
                 base_model_l1_outputs,
                 nb_features,
                 nb_input_slices=5,
                 dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.base_model = base_model
        self.nb_features = nb_features
        self.nb_input_slices = nb_input_slices
        self.base_model_features = base_model_features

        self.l1 = nn.Conv2d(nb_input_slices, base_model_l1_outputs, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(base_model_l1_outputs)
        self.fc = nn.Linear(base_model_features*2, nb_features)

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        self.base_model.train()
        self.base_model.requires_grad = True
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs, output_per_pixel=False):
        res = []
        x = inputs
        x = self.l1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)

        if output_per_pixel:
            res.append(F.conv2d(torch.cat([x, x], dim=1), self.fc.weight[:, :, None, None], self.fc.bias))

        avg_pool = F.avg_pool2d(x, x.shape[2:])
        max_pool = F.max_pool2d(x, x.shape[2:])
        avg_max_pool = torch.cat((avg_pool, max_pool), 1)
        x = avg_max_pool.view(avg_max_pool.size(0), -1)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, self.training)

        out = self.fc(x)

        if res:
            res.append(out)
            return res
        else:
            return out


def classification_model_resnet34_combine_last(**kwargs):
    base_model = pretrainedmodels.resnet34()
    return ClassificationModelResnetCombineLast(
        base_model,
        base_model_features=512,
        nb_features=6,
        base_model_l1_outputs=64,
        **kwargs)


def classification_model_resnet34_combine_first(**kwargs):
    base_model = pretrainedmodels.resnet34()
    return ClassificationModelResnetCombineFirst(
        base_model,
        base_model_features=512,
        nb_features=6,
        base_model_l1_outputs=64,
        **kwargs)
