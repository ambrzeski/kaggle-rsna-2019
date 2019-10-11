import torch
import torch.nn as nn
import torch.nn.functional as F

import pretrainedmodels
import torchvision

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
            res.append(F.conv2d(x, self.fc.weight[:, :, None, None], self.fc.bias))

        x, heatmap = self.gwap(x, output_heatmap=True)
        if output_heatmap:
            res.append(heatmap)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, self.training)
        out = self.fc(x)

        if res:
            res.append(out)
            return res
        else:
            return out


class ClassificationModelDPN(nn.Module):
    def __init__(self, base_model, base_model_features, base_model_l1_outputs, nb_features, dropout=0.5,
                 use_gwap=True, gwap_scale=2.0, nb_windows_conv=-1):
        super().__init__()
        self.dropout = dropout
        self.base_model = base_model
        self.nb_features = nb_features
        self.use_gwap = use_gwap
        self.nb_windows_conv = nb_windows_conv

        if nb_windows_conv == -1:
            self.l1_4 = nn.Conv2d(1, base_model_l1_outputs, kernel_size=7, stride=2, padding=3, bias=False)
            self.base_model.features[0].conv = self.l1_4
        else:
            self.l1_4 = nn.Sequential(
                nn.Conv2d(1, nb_windows_conv, kernel_size=1, bias=True),
                nn.ReLU(),
                nn.Conv2d(nb_windows_conv, base_model_l1_outputs, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            self.base_model.features[0] = self.l1_4

        if use_gwap:
            self.gwap = GWAP(base_model_features, scale=gwap_scale)
            self.fc = nn.Linear(base_model_features, nb_features)
        else:
            self.fc = nn.Linear(base_model_features*2, nb_features)

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        for param in self.l1_4.parameters():
            param.requires_grad = True

    def unfreeze_encoder(self):
        self.base_model.requires_grad = True
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs, output_per_pixel=False, output_heatmap=False):
        x = self.base_model.features(inputs)

        res = []

        if output_per_pixel:
            res.append(F.conv2d(x, self.fc.weight[:, :, None, None], self.fc.bias))

        heatmap = None
        if self.use_gwap:
            x, heatmap = self.gwap(x, output_heatmap=True)
        else:
            avg_pool = F.avg_pool2d(x, x.shape[2:])
            max_pool = F.max_pool2d(x, x.shape[2:])
            avg_max_pool = torch.cat((avg_pool, max_pool), 1)
            x = avg_max_pool.view(avg_max_pool.size(0), -1)

        if output_heatmap:
            res.append(heatmap)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, self.training)

        out = self.fc(x)

        if res:
            res.append(out)
            return res
        else:
            return out


class ClassificationModelMobilenet(nn.Module):
    def __init__(self, base_model, base_model_features, base_model_l1_outputs, nb_features, dropout=0.5,
                 use_gwap=True, gwap_scale=2.0):
        super().__init__()
        self.dropout = dropout
        self.base_model = base_model
        self.nb_features = nb_features
        self.use_gwap = use_gwap

        self.l1 = nn.Sequential(
            nn.Conv2d(1, base_model_l1_outputs, kernel_size=3, stride=2, padding=1, groups=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.l1_4 = nn.Conv2d(1, base_model_l1_outputs, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.features[0] = self.l1_4

        if use_gwap:
            self.gwap = GWAP(base_model_features, scale=gwap_scale)
            self.fc = nn.Linear(base_model_features, nb_features)
        else:
            self.fc = nn.Linear(base_model_features*2, nb_features)

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        for param in self.l1_4.parameters():
            param.requires_grad = True

    def unfreeze_encoder(self):
        self.base_model.requires_grad = True
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs, output_per_pixel=False, output_heatmap=False):
        x = self.base_model.features(inputs)

        res = []

        if output_per_pixel:
            res.append(F.conv2d(x, self.fc.weight[:, :, None, None], self.fc.bias))

        heatmap = None
        if self.use_gwap:
            x, heatmap = self.gwap(x, output_heatmap=True)
        else:
            avg_pool = F.avg_pool2d(x, x.shape[2:])
            max_pool = F.max_pool2d(x, x.shape[2:])
            avg_max_pool = torch.cat((avg_pool, max_pool), 1)
            x = avg_max_pool.view(avg_max_pool.size(0), -1)

        if output_heatmap:
            res.append(heatmap)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, self.training)

        out = self.fc(x)

        if res:
            res.append(out)
            return res
        else:
            return out


class ClassificationModelResnet(nn.Module):
    def __init__(self, base_model, base_model_features, base_model_l1_outputs, nb_features, dropout=0.5,
                 use_gwap=True, gwap_scale=2.0, nb_windows_conv=-1):
        super().__init__()
        self.dropout = dropout
        self.base_model = base_model
        self.nb_features = nb_features
        self.use_gwap = use_gwap
        self.nb_windows_conv = nb_windows_conv

        if nb_windows_conv > 0:
            self.windows_conv = nn.Conv2d(1, nb_windows_conv, kernel_size=1, stride=1, padding=1, groups=1, bias=True)
            self.l1 = nn.Conv2d(nb_windows_conv, base_model_l1_outputs, kernel_size=5, stride=2, padding=2, bias=True)
            torch.nn.init.uniform(self.windows_conv.weight, 2, 10)
            torch.nn.init.uniform(self.windows_conv.weight, -1, 1)
        else:
            self.l1 = nn.Conv2d(1, base_model_l1_outputs, kernel_size=7, stride=2, padding=3, bias=True)

        if use_gwap:
            self.gwap = GWAP(base_model_features, scale=gwap_scale)
            self.fc = nn.Linear(base_model_features, nb_features)
        else:
            self.fc = nn.Linear(base_model_features*2, nb_features)

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        self.base_model.requires_grad = True
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs, output_per_pixel=False, output_heatmap=False):
        res = []
        x = inputs
        if self.nb_windows_conv > 0:
            x = self.windows_conv(x)
            x = F.relu6(x)
            x = self.l1(x)  # should we put bn after?
            x = F.relu6(x)
        else:
            x = self.l1(x)
            x = torch.relu(x)
            # x = self.base_model.bn1(x)

        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)

        if output_per_pixel:
            res.append(F.conv2d(x, self.fc.weight[:, :, None, None], self.fc.bias))

        heatmap = None
        if self.use_gwap:
            x, heatmap = self.gwap(x, output_heatmap=True)
        else:
            avg_pool = F.avg_pool2d(x, x.shape[2:])
            max_pool = F.max_pool2d(x, x.shape[2:])
            avg_max_pool = torch.cat((avg_pool, max_pool), 1)
            x = avg_max_pool.view(avg_max_pool.size(0), -1)

        if output_heatmap:
            res.append(heatmap)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, self.training)

        out = self.fc(x)

        if res:
            res.append(out)
            return res
        else:
            return out


def classification_model_se_resnext50_gwap(**kwargs):
    base_model = pretrainedmodels.se_resnext50_32x4d()
    return ClassificationModelResNextGWAP(base_model, nb_features=6, **kwargs)


def classification_model_dpn92(**kwargs):
    base_model = pretrainedmodels.dpn92()
    return ClassificationModelDPN(base_model,
                                  base_model_features=2688,
                                  nb_features=6,
                                  base_model_l1_outputs=64,
                                  **kwargs)


def classification_model_dpn68b(**kwargs):
    base_model = pretrainedmodels.dpn68b()
    return ClassificationModelDPN(base_model,
                                  base_model_features=832,
                                  nb_features=6,
                                  base_model_l1_outputs=10,
                                  **kwargs)


def classification_model_mobilenet_v2(**kwargs):
    base_model = torchvision.models.mobilenet_v2(pretrained=True)
    return ClassificationModelMobilenet(base_model,
                                  base_model_features=1280,
                                  nb_features=6,
                                  base_model_l1_outputs=32,
                                  **kwargs)


def classification_model_resnet34(**kwargs):
    base_model = pretrainedmodels.resnet34()
    return ClassificationModelResnet(
        base_model,
        base_model_features=512,
        nb_features=6,
        base_model_l1_outputs=64,
        **kwargs)
