import torch
import torch.nn as nn
import torch.nn.functional as F

import pretrainedmodels


class GWAP(nn.Module):
    def __init__(self, channels, scale=1.0, extra_layer=None):
        super().__init__()
        self.scale = scale
        if extra_layer is not None:
            self.w1 = nn.Sequential(
                nn.Conv2d(channels, extra_layer, kernel_size=1, bias=True),
                nn.ReLU(),
                nn.Conv2d(extra_layer, 1, kernel_size=1, bias=True)
            )
        else:
            self.w1 = nn.Conv2d(channels, 1, kernel_size=1, bias=True)

    def forward(self, inputs, output_heatmap=False):
        inputs = inputs[:, :, 1:-1, 1:-1]  # discard borders, TODO: check it's ok for the task

        x = self.w1(inputs)
        m = torch.exp(self.scale*torch.sigmoid(x))
        a = m / torch.sum(m, dim=(2, 3), keepdim=True)

        x = a * inputs
        gwap = torch.sum(x, dim=(2, 3))
        if output_heatmap:
            return gwap, a
        else:
            return gwap


class ClassificationModelResNextGWAP(nn.Module):
    def __init__(self, base_model, nb_features, dropout=0.5, gwap_channels=2048):
        super().__init__()
        self.base_model = base_model
        self.dropout = dropout
        self.nb_features = nb_features

        self.l1_4 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.layer0[0] = self.l1_4

        self.gwap = GWAP(gwap_channels, scale=1.5)
        self.fc = nn.Linear(gwap_channels, nb_features)

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.requires_grad = False

        for param in self.l1_4.parameters():
            param.requires_grad = True

    def unfreeze_encoder(self):
        self.base_model.requires_grad = True
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs, output_heatmap=False, output_per_pixel=False):
        x = self.base_model.layer0(inputs)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        res = []

        if output_per_pixel:
            # conv = nn.Conv2d(x.shape[1], self.nb_features, kernel_size=1)
            # print(self.fc.weight.shape, self.fc.bias.shape)
            # print(conv.weight.shape, conv.bias.shape)
            # conv.load_state_dict({"weight": self.fc.weight[:, :, None, None],
            #                       "bias": self.fc.bias})
            res.append(F.conv2d(x, self.fc.weight[:, :, None, None], self.fc.bias))

        x, heatmap = self.gwap(x, output_heatmap=True)
        if output_heatmap:
            res.append(heatmap)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, self.training)
        out = self.fc(x)
        res.append(out)

        return res


def classification_model_se_resnext50_gwap(**kwargs):
    base_model = pretrainedmodels.se_resnext50_32x4d()
    return ClassificationModelResNextGWAP(base_model, nb_features=6, **kwargs)

