import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import pretrainedmodels
import efficientnet_pytorch


class CombineLast3DWrapper(nn.Module):
    def __init__(self,
                 base_model,
                 base_model_features,
                 combine_slices=5,
                 combine_conv_features=256
                 ):
        super().__init__()
        self.combine_slices = combine_slices
        self.base_model = base_model
        self.base_model_features = base_model_features

        self.combine_conv_features = combine_conv_features
        self.combine_conv = nn.Conv3d(base_model_features, self.combine_conv_features,
                                      kernel_size=(combine_slices, 1, 1),
                                      padding=((combine_slices-1)//2, 0, 0))
        # self.combine_conv.weight.data.fill_(0.1)
        self.combine_conv.bias.data.fill_(0.0)

    def forward(self, inputs, output_per_pixel=False):
        """
        :param inputs: BxSxHxW
        :param output_per_pixel:
        :return:
        """
        res = []
        batch_size = inputs.shape[0]
        assert batch_size == 1
        nb_input_slices = inputs.shape[1]

        x = inputs.view(batch_size*nb_input_slices, 1, inputs.shape[2], inputs.shape[3])

        x = self.base_model(x)
        base_model_features = x.shape[1]
        x = x.view(batch_size, nb_input_slices, base_model_features, x.shape[2], x.shape[3])
        x = x.permute((0, 2, 1, 3, 4))  # BxCxSxHxW
        x = self.combine_conv(x)

        # TODO: seems to work worse with relu here, need more testing
        # x = torch.relu(x)

        if output_per_pixel:
            res.append(F.conv3d(torch.cat([x, x], dim=1), self.fc.weight[:, :, None, None], self.fc.bias))

        # x: BxCxSxHxW
        avg_pool = F.avg_pool3d(x, (1,)+x.shape[3:])
        max_pool = F.max_pool3d(x, (1,)+x.shape[3:])
        avg_max_pool = torch.cat((avg_pool, max_pool), 1)
        # x: Bx2CxSx1x1
        out = avg_max_pool[:, :, :, 0, 0]

        if res:
            res.append(out)
            return res
        else:
            return out


def test_combine_last_3d_wrapper():
    base_model = lambda x: x.repeat(1, 2, 1, 1)
    m = CombineLast3DWrapper(base_model=base_model, base_model_features=2, combine_slices=3, combine_conv_features=8)

    batch_size = 1
    w = 4
    h = 5
    slices = 3

    x = torch.zeros((batch_size, slices, h, w))

    for i in range(slices):
        x[:, i, :, :] += i

    for i in range(h):
        x[:, :, i, :] += i*0.1

    for i in range(w):
        x[:, :, :, i] += i*0.01

    pred = m(x)
    print(pred)


class ClassificationModelResnetCombineLast(nn.Module):
    def __init__(self,
                 base_model,
                 base_model_features,
                 base_model_l1_outputs,
                 nb_features,
                 combine_slices=5,
                 combine_conv_features=256,
                 dropout=0.5):
        super().__init__()
        self.combine_slices = combine_slices
        self.dropout = dropout
        self.base_model = base_model
        self.nb_features = nb_features
        self.base_model_features = base_model_features

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.l1 = nn.Conv2d(1, base_model_l1_outputs, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_model_l1_outputs)

        self.combine_conv_features = combine_conv_features
        self.combine_conv = nn.Conv3d(base_model_features, self.combine_conv_features,
                                      kernel_size=(combine_slices, 1, 1),
                                      # padding=((combine_slices-1)//2, 0, 0)
                                      )
        self.fc = nn.Conv1d(self.combine_conv_features*2, nb_features, kernel_size=1)

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        self.base_model.requires_grad = True
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs, output_per_pixel=False):
        """
        :param inputs: BxSxHxW
        :param output_per_pixel:
        :return:
        """
        res = []
        batch_size = inputs.shape[0]
        nb_input_slices = inputs.shape[1]

        x = inputs.view(batch_size*nb_input_slices, 1, inputs.shape[2], inputs.shape[3])
        x = self.l1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)  # BSxCxHxW

        base_model_features = x.shape[1]
        x = x.view(batch_size, nb_input_slices, base_model_features, x.shape[2], x.shape[3])  # BxSxCxHxW
        x = x.permute((0, 2, 1, 3, 4))  # BxCxSxHxW
        x = self.combine_conv(x)

        if output_per_pixel:
            res.append(F.conv3d(torch.cat([x, x], dim=1), self.fc.weight[:, :, None, None], self.fc.bias))

        # x: BxCxSxHxW
        avg_pool = F.avg_pool3d(x, (1,)+x.shape[3:])
        max_pool = F.max_pool3d(x, (1,)+x.shape[3:])
        avg_max_pool = torch.cat((avg_pool, max_pool), 1)
        # x: Bx2CxSx1x1
        x = avg_max_pool[:, :, :, 0, 0]

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, self.training)

        out = self.fc(x)  # BxCxS
        # x: Bx2CxS
        # out = out[None, :, :]
        out = out.permute(0, 2, 1)  # BxSxC

        if res:
            res.append(out)
            return res
        else:
            return out


class ClassificationModelENetCombineLast(nn.Module):
    def __init__(self,
                 base_model,
                 base_model_features,
                 base_model_l1_outputs,
                 nb_features,
                 combine_slices=5,
                 combine_conv_features=256,
                 dropout=0.5):
        super().__init__()
        self.combine_slices = combine_slices
        self.dropout = dropout
        self.base_model = base_model
        self.nb_features = nb_features
        self.base_model_features = base_model_features

        self.base_model._conv_stem = nn.Conv2d(1, base_model_l1_outputs, kernel_size=3, stride=2, bias=False)

        self.combine_conv_features = combine_conv_features
        self.combine_conv = nn.Conv3d(base_model_features, self.combine_conv_features,
                                      kernel_size=(combine_slices, 1, 1),
                                      # padding=((combine_slices-1)//2, 0, 0)
                                      )
        self.fc = nn.Conv1d(self.combine_conv_features*2, nb_features, kernel_size=1)

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.base_model._conv_stem.requires_grad = True
        self.base_model._bn0.requires_grad = True
        self.base_model._bn0.train()

    def unfreeze_encoder(self):
        self.base_model.requires_grad = True
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs, output_per_pixel=False):
        """
        :param inputs: BxSxHxW
        :param output_per_pixel:
        :return:
        """
        res = []
        batch_size = inputs.shape[0]
        nb_input_slices = inputs.shape[1]

        x = inputs.view(batch_size*nb_input_slices, 1, inputs.shape[2], inputs.shape[3])
        x = self.base_model.extract_features(x)  # BSxCxHxW

        base_model_features = x.shape[1]
        x = x.view(batch_size, nb_input_slices, base_model_features, x.shape[2], x.shape[3])  # BxSxCxHxW
        x = x.permute((0, 2, 1, 3, 4))  # BxCxSxHxW
        x = self.combine_conv(x)

        if output_per_pixel:
            res.append(F.conv3d(torch.cat([x, x], dim=1), self.fc.weight[:, :, None, None], self.fc.bias))

        # x: BxCxSxHxW
        avg_pool = F.avg_pool3d(x, (1,)+x.shape[3:])
        max_pool = F.max_pool3d(x, (1,)+x.shape[3:])
        avg_max_pool = torch.cat((avg_pool, max_pool), 1)
        # x: Bx2CxSx1x1
        x = avg_max_pool[:, :, :, 0, 0]

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, self.training)

        out = self.fc(x)  # BxCxS
        # x: Bx2CxS
        # out = out[None, :, :]
        out = out.permute(0, 2, 1)  # BxSxC

        if res:
            res.append(out)
            return res
        else:
            return out


class ClassificationModelResnetCombineL3(nn.Module):
    def __init__(self,
                 base_model,
                 base_model_l3_features,
                 base_model_features,
                 base_model_l1_outputs,
                 nb_features,
                 combine_slices=5,
                 combine_conv_features=256,
                 dropout=0.5,
                 use_wso=False
                 ):
        super().__init__()
        self.combine_slices = combine_slices
        self.dropout = dropout
        self.base_model = base_model
        self.nb_features = nb_features
        self.base_model_features = base_model_features

        self.use_wso = use_wso

        if self.use_wso:
            self.wso = WSO()
            self.l1 = nn.Conv2d(4, base_model_l1_outputs, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.l1 = nn.Conv2d(1, base_model_l1_outputs, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_model_l1_outputs)

        self.combine_conv_features = combine_conv_features
        self.combine_conv = nn.Conv3d(base_model_l3_features, base_model_l3_features,
                                      kernel_size=(combine_slices, 1, 1),
                                      # padding=((combine_slices-1)//2, 0, 0)
                                      )
        self.fc = nn.Conv1d(base_model_features*2, nb_features, kernel_size=1)

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        if self.use_wso:
            self.wso.requires_grad = False

    def unfreeze_encoder(self):
        self.base_model.requires_grad = True
        for param in self.base_model.parameters():
            param.requires_grad = True

        if self.use_wso:
            self.wso.requires_grad = True

    def forward(self, inputs, output_per_pixel=False):
        """
        :param inputs: BxSxHxW
        :param output_per_pixel:
        :return:
        """
        res = []
        batch_size = inputs.shape[0]
        nb_input_slices = inputs.shape[1]

        x = inputs.view(batch_size*nb_input_slices, 1, inputs.shape[2], inputs.shape[3])
        if self.use_wso:
            x = self.wso(x)
        x = self.l1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)  # BSxCxHxW

        base_model_features = x.shape[1]
        x = x.view(batch_size, nb_input_slices, base_model_features, x.shape[2], x.shape[3])  # BxSxCxHxW
        x = x.permute((0, 2, 1, 3, 4))  # BxCxSxHxW
        x = self.combine_conv(x)
        x = x.permute((0, 2, 1, 3, 4))  # BxSxCxHxW
        nb_output_slices = x.shape[1]
        x = x.reshape((batch_size*nb_output_slices, x.shape[2], x.shape[3], x.shape[4]))  # BSxCxHxW
        x = self.base_model.layer4(x)
        x = x.reshape((batch_size, nb_output_slices, x.shape[1], x.shape[2], x.shape[3]))  # BxSxCxHxW
        x = x.permute((0, 2, 1, 3, 4))  # BxCxSxHxW

        # if output_per_pixel:
        #     res.append(F.conv3d(torch.cat([x, x], dim=1), self.fc.weight[:, :, None, None], self.fc.bias))

        # x: BxCxSxHxW
        avg_pool = F.avg_pool3d(x, (1,)+x.shape[3:])
        max_pool = F.max_pool3d(x, (1,)+x.shape[3:])
        avg_max_pool = torch.cat((avg_pool, max_pool), 1)
        # x: Bx2CxSx1x1
        x = avg_max_pool[:, :, :, 0, 0]

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, self.training)

        out = self.fc(x)  # BxCxS
        # x: Bx2CxS
        # out = out[None, :, :]
        out = out.permute(0, 2, 1)  # BxSxC

        # if res:
        #     res.append(out)
        #     return res
        # else:
        return out



class ClassificationModelDPN(nn.Module):
    def __init__(self,
                 base_model,
                 base_model_features,
                 base_model_l1_outputs,
                 nb_features,
                 combine_slices=5,
                 combine_conv_features=256,
                 dropout=0.5):
        super().__init__()
        self.combine_slices = combine_slices
        self.dropout = dropout
        self.base_model = base_model
        self.nb_features = nb_features
        self.base_model_features = base_model_features

        self.l1_4 = nn.Sequential(
            nn.Conv2d(1, base_model_l1_outputs, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_model_l1_outputs, eps=0.001, momentum=0.02),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.base_model.features[0] = self.l1_4

        self.combine_conv_features = combine_conv_features
        self.combine_conv = nn.Conv3d(base_model_features, self.combine_conv_features,
                                      kernel_size=(combine_slices, 1, 1),
                                      # padding=((combine_slices-1)//2, 0, 0)
                                      )
        self.fc = nn.Conv1d(self.combine_conv_features*2, nb_features, kernel_size=1)

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.l1_4.train()
        self.l1_4.requires_grad = True

    def unfreeze_encoder(self):
        self.base_model.requires_grad = True
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs, output_per_pixel=False):
        """
        :param inputs: BxSxHxW
        :param output_per_pixel:
        :return:
        """
        res = []
        batch_size = inputs.shape[0]
        nb_input_slices = inputs.shape[1]

        x = inputs.view(batch_size*nb_input_slices, 1, inputs.shape[2], inputs.shape[3])
        x = self.base_model.features(x)
        base_model_features = x.shape[1]
        x = x.view(batch_size, nb_input_slices, base_model_features, x.shape[2], x.shape[3])  # BxSxCxHxW
        x = x.permute((0, 2, 1, 3, 4))  # BxCxSxHxW
        x = self.combine_conv(x)

        if output_per_pixel:
            res.append(F.conv3d(torch.cat([x, x], dim=1), self.fc.weight[:, :, None, None], self.fc.bias))

        # x: BxCxSxHxW
        avg_pool = F.avg_pool3d(x, (1,)+x.shape[3:])
        max_pool = F.max_pool3d(x, (1,)+x.shape[3:])
        avg_max_pool = torch.cat((avg_pool, max_pool), 1)
        # x: Bx2CxSx1x1
        x = avg_max_pool[:, :, :, 0, 0]

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, self.training)

        out = self.fc(x)  # BxCxS
        # x: Bx2CxS
        # out = out[None, :, :]
        out = out.permute(0, 2, 1)  # BxSxC

        if res:
            res.append(out)
            return res
        else:
            return out

import pytorchcv.models.airnet
import pytorchcv.models.sepreresnet


class ClassificationModelAirnet(nn.Module):
    def __init__(self,
                 base_model_features,
                 base_model_l1_outputs,
                 nb_features,
                 combine_slices=5,
                 combine_conv_features=128,
                 dropout=0.0):
        super().__init__()
        self.combine_slices = combine_slices
        self.dropout = dropout
        self.nb_features = nb_features
        self.base_model_features = base_model_features

        self.base_model = pytorchcv.models.airnet.airnet50_1x64d_r16(pretrained=True)
        self.base_model.features[0].conv1.conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        self.combine_conv_features = combine_conv_features
        self.combine_conv = nn.Conv3d(base_model_features, self.combine_conv_features,
                                      kernel_size=(combine_slices, 1, 1),
                                      # padding=((combine_slices-1)//2, 0, 0)
                                      )
        self.fc = nn.Conv1d(self.combine_conv_features*2, nb_features, kernel_size=1)

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.base_model.features[0].conv1.conv.requires_grad = True
        self.base_model.features[0].conv1.bn.requires_grad = True
        self.base_model.features[0].conv1.bn.train()

    def unfreeze_encoder(self):
        self.base_model.requires_grad = True
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs, output_per_pixel=False):
        """
        :param inputs: BxSxHxW
        :param output_per_pixel:
        :return:
        """
        res = []
        batch_size = inputs.shape[0]
        nb_input_slices = inputs.shape[1]

        x = inputs.view(batch_size*nb_input_slices, 1, inputs.shape[2], inputs.shape[3])
        x = self.base_model.features(x)
        base_model_features = x.shape[1]
        x = x.view(batch_size, nb_input_slices, base_model_features, x.shape[2], x.shape[3])  # BxSxCxHxW
        x = x.permute((0, 2, 1, 3, 4))  # BxCxSxHxW
        x = self.combine_conv(x)

        if output_per_pixel:
            res.append(F.conv3d(torch.cat([x, x], dim=1), self.fc.weight[:, :, None, None], self.fc.bias))

        # x: BxCxSxHxW
        avg_pool = F.avg_pool3d(x, (1,)+x.shape[3:])
        max_pool = F.max_pool3d(x, (1,)+x.shape[3:])
        avg_max_pool = torch.cat((avg_pool, max_pool), 1)
        # x: Bx2CxSx1x1
        x = avg_max_pool[:, :, :, 0, 0]

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, self.training)

        out = self.fc(x)  # BxCxS
        # x: Bx2CxS
        # out = out[None, :, :]
        out = out.permute(0, 2, 1)  # BxSxC

        if res:
            res.append(out)
            return res
        else:
            return out



class WSO(nn.Module):
    def __init__(self,  windows=None, U=255., eps=8.):
        super(WSO, self).__init__()

        if windows is None:
            windows = OrderedDict({
                'brain': {'W': 80, 'L': 40},
                'subdural': {'W': 215, 'L': 75},
                'bony': {'W': 2800, 'L': 600},
                'tissue': {'W': 375, 'L': 40},
            })

        self.U = U
        self.eps = eps

        self.windows = windows
        self.conv1x1 = nn.Conv2d(1, len(windows), kernel_size=1, stride=1, padding=0)
        weights, bias = self._get_window_params()
        self.conv1x1.weight.data = weights.reshape(self.conv1x1.weight.data.shape)
        self.conv1x1.bias.data = bias.reshape(self.conv1x1.bias.data.shape)

    def _get_window_params(self):
        weight = []
        bias = []

        def get_init_conv_params_sigmoid(ww, wl, smooth=1., upbound_value=255.):
            w = 2. / ww * math.log(upbound_value / smooth - 1.)
            b = -2. * wl / ww * math.log(upbound_value / smooth - 1.)
            return w, b

        for _, window in self.windows.items():
            ww, wl = window["W"], window["L"]
            w, b = get_init_conv_params_sigmoid(ww, wl, self.eps, self.U)
            weight.append(w)
            bias.append(b)
        weight = torch.as_tensor(weight)
        bias = torch.as_tensor(bias)
        # print(weight, bias)
        return weight, bias

    def forward(self, x):
        x = self.conv1x1(x)
        x = torch.sigmoid(x)
        return x


def check_wso():
    from rsna19.data import dataset
    import albumentations.pytorch
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    wso = WSO()

    dataset_valid = dataset.IntracranialDataset(
        csv_file='5fold.csv',
        folds=[0],
        preprocess_func=albumentations.pytorch.ToTensorV2(),
    )
    batch_size = 2
    data_loader = DataLoader(dataset_valid,
                             shuffle=False,
                             num_workers=16,
                             batch_size=batch_size)

    for data in data_loader:
        img = data['image'].float().cpu()

        windowed_img = wso(img).detach().numpy()

        fig, ax = plt.subplots(4, 1)

        for batch in range(batch_size):
            for j in range(4):
                # for k in range(4):
                    ax[j].imshow(windowed_img[batch, j], cmap='gray')

        plt.show()



def classification_model_resnet34_combine_last(**kwargs):
    base_model = pretrainedmodels.resnet34()
    return ClassificationModelResnetCombineLast(
        base_model,
        base_model_features=512,
        nb_features=6,
        base_model_l1_outputs=64,
        **kwargs)


def classification_model_resnext50_combine_last(**kwargs):
    base_model = pretrainedmodels.se_resnext50_32x4d()
    return ClassificationModelResnetCombineLast(
        base_model,
        base_model_features=2048,
        nb_features=6,
        base_model_l1_outputs=64,
        combine_conv_features=512,
        **kwargs)


def classification_model_enet_b0_combine_last(**kwargs):
    base_model = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b0')
    return ClassificationModelENetCombineLast(
        base_model,
        base_model_features=1280,
        nb_features=6,
        base_model_l1_outputs=32,
        combine_conv_features=512,
        **kwargs)


def classification_model_enet_b2_combine_last(**kwargs):
    base_model = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b2')
    return ClassificationModelENetCombineLast(
        base_model,
        base_model_features=1408,
        nb_features=6,
        base_model_l1_outputs=32,
        combine_conv_features=512,
        **kwargs)


def classification_model_resnet34_combine_l3(**kwargs):
    base_model = pretrainedmodels.resnet34()
    return ClassificationModelResnetCombineL3(
        base_model,
        base_model_features=512,
        base_model_l3_features=256,
        nb_features=6,
        base_model_l1_outputs=64,
        **kwargs)


def classification_model_dpn68_combine_last(**kwargs):
    base_model = pretrainedmodels.dpn68b()
    return ClassificationModelDPN(base_model,
                                  base_model_features=832,
                                  nb_features=6,
                                  base_model_l1_outputs=10,
                                  **kwargs)

def classification_model_airnet50(**kwargs):
    return ClassificationModelAirnet(base_model_features=2048,
                                  nb_features=6,
                                  base_model_l1_outputs=10,
                                  **kwargs)


if __name__ == '__main__':
    # test_combine_last_3d_wrapper()
    check_wso()


