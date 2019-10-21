import torch
import torch.nn as nn
import torch.nn.functional as F

import pretrainedmodels


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
        x = self.base_model.maxpool(x)

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


def classification_model_resnet34_combine_last(**kwargs):
    base_model = pretrainedmodels.resnet34()
    return ClassificationModelResnetCombineLast(
        base_model,
        base_model_features=512,
        nb_features=6,
        base_model_l1_outputs=64,
        **kwargs)


if __name__ == '__main__':
    test_combine_last_3d_wrapper()
