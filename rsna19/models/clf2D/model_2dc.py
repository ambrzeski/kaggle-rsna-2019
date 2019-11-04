import torch
import torch.nn as nn
import torch.nn.functional as F

import pretrainedmodels
import pytorchcv.models.airnet
import pytorchcv.models.airnext
import pytorchcv.models.sepreresnet


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


class ClassificationModelResnetCombineLastVariable(nn.Module):
    def __init__(self,
                 base_model,
                 base_model_features,
                 base_model_l1_outputs,
                 nb_features,
                 nb_input_slices=5,
                 combine_conv_features=256,
                 dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.base_model = base_model
        self.nb_features = nb_features
        self.nb_input_slices = nb_input_slices
        self.base_model_features = base_model_features
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.l1 = nn.Conv2d(1, base_model_l1_outputs, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_model_l1_outputs)
        self.combine_conv = nn.Conv2d(base_model_features * nb_input_slices, 256, kernel_size=1)

        self.combine_conv_features = combine_conv_features
        self.combine_conv = nn.Conv3d(base_model_features, self.combine_conv_features,
                                      kernel_size=(nb_input_slices, 1, 1))
        center_slice = (nb_input_slices - 1) // 2
        nn.init.zeros_(self.combine_conv.bias.data)
        with torch.no_grad():
            self.combine_conv.weight.data[:, :, :center_slice, :, :] = 0
            self.combine_conv.weight.data[:, :, center_slice+1:, :, :] = 0

        print(self.combine_conv.weight.data.shape, self.combine_conv.bias.data.shape)
        self.fc = nn.Linear(combine_conv_features*2, nb_features)

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

    def freeze_encoder_full(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.l1.requires_grad = False
        self.bn1.requires_grad = False

    def freeze_bn(self):
        def set_bn_eval(m):
            if 'BatchNorm' in m.__class__.__name__:
                # print('eval', m)
                m.eval()
                m.requires_grad = False
        self.apply(set_bn_eval)

    def unfreeze_bn(self):
        def set_bn_train(m):
            if 'BatchNorm' in m.__class__.__name__:
                m.requires_grad = True
        self.apply(set_bn_train)

    def unfreeze_encoder(self):
        self.base_model.train()
        self.base_model.requires_grad = True
        for param in self.base_model.parameters():
            param.requires_grad = True
        self.l1.requires_grad = True
        self.bn1.requires_grad = True

    def forward(self, inputs, output_per_pixel=False, output_before_combine_slices=False, train_last_layers_only=False):
        res = []
        batch_size = inputs.shape[0]
        nb_input_slices = inputs.shape[1]

        x = inputs

        if not train_last_layers_only:
            x = x.view(batch_size*nb_input_slices, 1, inputs.shape[2], inputs.shape[3])
            x = self.l1(x)
            x = self.bn1(x)
            # TODO: batch norm here may still help when cdf used
            x = torch.relu(x)
            x = self.maxpool(x)

            x = self.base_model.layer1(x)
            x = self.base_model.layer2(x)
            x = self.base_model.layer3(x)
            x = self.base_model.layer4(x)

            base_model_features = x.shape[1]
            x = x.view(batch_size, nb_input_slices, base_model_features, x.shape[2], x.shape[3])  # BxSxCxHxW

        if output_before_combine_slices:
            return x

        x = x.permute((0, 2, 1, 3, 4))  # BxCxSxHxW
        # x = self.combine_conv(x)  # BxCx1xHxW
        slice_offset = (self.nb_input_slices - nb_input_slices) // 2
        x = F.conv3d(x,
                     self.combine_conv.weight[:, :, slice_offset:slice_offset+nb_input_slices, :, :],
                     self.combine_conv.bias)

        x = x.view(batch_size, self.combine_conv_features, x.shape[3], x.shape[4])

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



class ClassificationModelDPNCombineLastVariable(nn.Module):
    def __init__(self,
                 base_model,
                 base_model_features,
                 base_model_l1_outputs,
                 nb_features,
                 nb_input_slices=5,
                 combine_conv_features=256,
                 dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.base_model = base_model
        self.nb_features = nb_features
        self.nb_input_slices = nb_input_slices
        self.base_model_features = base_model_features
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.base_model.features[0].conv = nn.Conv2d(1, base_model_l1_outputs, kernel_size=3, stride=2, padding=1, bias=False)
        self.combine_conv = nn.Conv2d(base_model_features * nb_input_slices, 256, kernel_size=1)

        self.combine_conv_features = combine_conv_features
        self.combine_conv = nn.Conv3d(base_model_features, self.combine_conv_features,
                                      kernel_size=(nb_input_slices, 1, 1))
        center_slice = (nb_input_slices - 1) // 2
        nn.init.zeros_(self.combine_conv.bias.data)
        with torch.no_grad():
            self.combine_conv.weight.data[:, :, :center_slice, :, :] = 0
            self.combine_conv.weight.data[:, :, center_slice+1:, :, :] = 0

        print(self.combine_conv.weight.data.shape, self.combine_conv.bias.data.shape)
        self.fc = nn.Linear(combine_conv_features*2, nb_features)

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.base_model.features[0].conv.requires_grad = True
        self.base_model.features[0].bn.requires_grad = True
        self.base_model.features[0].bn.train()

    def unfreeze_encoder(self):
        self.base_model.train()
        self.base_model.requires_grad = True
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs, output_per_pixel=False, output_before_combine_slices=False, train_last_layers_only=False):
        res = []
        batch_size = inputs.shape[0]
        nb_input_slices = inputs.shape[1]

        x = inputs

        if not train_last_layers_only:
            x = x.view(batch_size*nb_input_slices, 1, inputs.shape[2], inputs.shape[3])
            x = self.base_model.features(x)
            base_model_features = x.shape[1]
            x = x.view(batch_size, nb_input_slices, base_model_features, x.shape[2], x.shape[3])  # BxSxCxHxW

        if output_before_combine_slices:
            return x

        x = x.permute((0, 2, 1, 3, 4))  # BxCxSxHxW
        # x = self.combine_conv(x)  # BxCx1xHxW
        slice_offset = (self.nb_input_slices - nb_input_slices) // 2
        x = F.conv3d(x,
                     self.combine_conv.weight[:, :, slice_offset:slice_offset+nb_input_slices, :, :],
                     self.combine_conv.bias)

        x = x.view(batch_size, self.combine_conv_features, x.shape[3], x.shape[4])

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


class BaseClassificationModel(nn.Module):
    def __init__(self,
                 base_model,
                 base_model_features,
                 nb_features=6,
                 nb_input_slices=5,
                 combine_conv_features=256,
                 dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.base_model = base_model
        self.nb_features = nb_features
        self.nb_input_slices = nb_input_slices
        self.base_model_features = base_model_features

        self.combine_conv_features = combine_conv_features
        self.combine_conv = nn.Conv3d(base_model_features, self.combine_conv_features,
                                      kernel_size=(nb_input_slices, 1, 1))
        center_slice = (nb_input_slices - 1) // 2
        nn.init.zeros_(self.combine_conv.bias.data)
        with torch.no_grad():
            self.combine_conv.weight.data[:, :, :center_slice, :, :] = 0
            self.combine_conv.weight.data[:, :, center_slice+1:, :, :] = 0

        print(self.combine_conv.weight.data.shape, self.combine_conv.bias.data.shape)
        self.fc = nn.Linear(combine_conv_features*2, nb_features)

    def freeze_encoder(self):
        self.base_model.freeze_encoder()

    def unfreeze_encoder(self):
        self.base_model.unfreeze_encoder()

    def forward(self, inputs, output_per_pixel=False):
        res = []
        batch_size = inputs.shape[0]
        nb_input_slices = inputs.shape[1]

        x = inputs
        x = x.view(batch_size * nb_input_slices, 1, inputs.shape[2], inputs.shape[3])
        x = self.base_model(x)
        base_model_features = x.shape[1]
        x = x.view(batch_size, nb_input_slices, base_model_features, x.shape[2], x.shape[3])  # BxSxCxHxW
        x = x.permute((0, 2, 1, 3, 4))  # BxCxSxHxW
        # x = self.combine_conv(x)  # BxCx1xHxW

        if nb_input_slices == self.nb_input_slices:
            x = self.combine_conv(x)  # BxCx1xHxW
        else:
            slice_offset = (self.nb_input_slices - nb_input_slices) // 2
            x = F.conv3d(x,
                         self.combine_conv.weight[:, :, slice_offset:slice_offset+nb_input_slices, :, :],
                         self.combine_conv.bias)

        x = x.view(batch_size, self.combine_conv_features, x.shape[3], x.shape[4])

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


class XCeptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = pretrainedmodels.xception()
        self.base_model.conv1 = nn.Conv2d(1, 32, 3, 2, 0, bias=False)

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.base_model.conv1.requires_grad = True
        self.base_model.bn1.requires_grad = True
        self.base_model.bn1.train()

    def unfreeze_encoder(self):
        self.base_model.train()
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        return self.base_model.features(inputs)


class NasNetMobileModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = torch.nn.ConstantPad2d(24, -1)
        self.base_model = pretrainedmodels.nasnetamobile(num_classes=1000)
        self.base_model.conv0[0] = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=0,
                                                  stride=2,
                                                  bias=False)

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.base_model.conv0[0].requires_grad = True
        self.base_model.conv0[1].requires_grad = True
        self.base_model.conv0[1].train()

    def unfreeze_encoder(self):
        self.base_model.train()
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        x = self.pad(inputs)
        return self.base_model.features(x)


class BnInceptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = torch.nn.ConstantPad2d(24, -1)
        self.base_model = pretrainedmodels.bninception()
        self.base_model.conv1_7x7_s2 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.base_model.conv1_7x7_s2.requires_grad = True
        self.base_model.conv1_7x7_s2_bn.requires_grad = True
        self.base_model.conv1_7x7_s2_bn.train()

    def unfreeze_encoder(self):
        self.base_model.train()
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        x = self.pad(inputs)
        return self.base_model.features(x)


class InceptionResnetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = torch.nn.ConstantPad2d(24, -1)
        self.base_model = pretrainedmodels.inceptionresnetv2()
        self.base_model.conv2d_1a.conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.base_model.conv2d_1a.conv.requires_grad = True
        self.base_model.conv2d_1a.bn.requires_grad = True
        self.base_model.conv2d_1a.bn.train()

    def unfreeze_encoder(self):
        self.base_model.train()
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        x = self.pad(inputs)
        return self.base_model.features(x)




class AirNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = pytorchcv.models.airnet.airnet50_1x64d_r16(pretrained=True)
        self.base_model.features[0].conv1.conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.base_model.features[0].conv1.conv.requires_grad = True
        self.base_model.features[0].conv1.bn.requires_grad = True
        self.base_model.features[0].conv1.bn.train()

    def unfreeze_encoder(self):
        self.base_model.train()
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        return self.base_model.features(inputs)


class AirNext(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = pytorchcv.models.airnext.airnext50_32x4d_r2(pretrained=True)
        self.base_model.features[0].conv1.conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.base_model.features[0].conv1.conv.requires_grad = True
        self.base_model.features[0].conv1.bn.requires_grad = True
        self.base_model.features[0].conv1.bn.train()

    def unfreeze_encoder(self):
        self.base_model.train()
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        return self.base_model.features(inputs)


class SE_PreResNet_BC_26b(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = pytorchcv.models.sepreresnet.sepreresnetbc26b(pretrained=True)
        self.base_model.features[0].conv = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.base_model.features[0].conv.requires_grad = True
        self.base_model.features[0].bn.requires_grad = True
        self.base_model.features[0].bn.train()

    def unfreeze_encoder(self):
        self.base_model.train()
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        return self.base_model.features(inputs)


class ResNext50(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = pretrainedmodels.se_resnext50_32x4d()
        self.base_model.layer0[0] = nn.Conv2d(1, 64, 3, stride=2, padding=1, bias=False)

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.base_model.layer0[0].requires_grad = True
        self.base_model.layer0[1].requires_grad = True
        self.base_model.layer0[1].train()

    def unfreeze_encoder(self):
        self.base_model.train()
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        return self.base_model.features(inputs)



def classification_model_resnet34_combine_last(**kwargs):
    base_model = pretrainedmodels.resnet34()
    return ClassificationModelResnetCombineLast(
        base_model,
        base_model_features=512,
        nb_features=6,
        base_model_l1_outputs=64,
        **kwargs)


def classification_model_resnet34_combine_last_var(**kwargs):
    base_model = pretrainedmodels.resnet34()
    return ClassificationModelResnetCombineLastVariable(
        base_model,
        base_model_features=512,
        nb_features=6,
        base_model_l1_outputs=64,
        **kwargs)


def classification_model_resnext50_combine_last_var(**kwargs):
    base_model = pretrainedmodels.se_resnext50_32x4d()
    return ClassificationModelResnetCombineLastVariable(
        base_model,
        base_model_features=2048,
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


def classification_model_resnet18_combine_last_var(**kwargs):
    base_model = pretrainedmodels.resnet18()
    return ClassificationModelResnetCombineLastVariable(
        base_model,
        base_model_features=512,
        nb_features=6,
        base_model_l1_outputs=64,
        **kwargs)


def classification_model_resnet50_combine_last_var(**kwargs):
    base_model = pretrainedmodels.resnet50()
    return ClassificationModelResnetCombineLastVariable(
        base_model,
        base_model_features=2048,
        nb_features=6,
        base_model_l1_outputs=64,
        **kwargs)


def classification_model_dpn68_combine_last_var(**kwargs):
    base_model = pretrainedmodels.dpn68b()
    return ClassificationModelDPNCombineLastVariable(
        base_model,
        base_model_features=832,
        nb_features=6,
        base_model_l1_outputs=10,
        **kwargs)


def classification_model_xception(**kwargs):
    base_model = XCeptionModel()
    return BaseClassificationModel(
        base_model,
        base_model_features=2048,
        nb_features=6,
        **kwargs)


def classification_model_nasnet_mobile(**kwargs):
    base_model = NasNetMobileModel()
    return BaseClassificationModel(
        base_model,
        base_model_features=1056,
        nb_features=6,
        **kwargs)


def classification_model_bninception(**kwargs):
    base_model = BnInceptionModel()
    return BaseClassificationModel(
        base_model,
        base_model_features=1024,
        nb_features=6,
        **kwargs)


def classification_model_inception_resnet_v2(**kwargs):
    base_model = InceptionResnetV2()
    return BaseClassificationModel(
        base_model,
        base_model_features=1536,
        nb_features=6,
        **kwargs)


def classification_model_airnet_50(**kwargs):
    base_model = AirNet()
    return BaseClassificationModel(
        base_model,
        base_model_features=2048,
        combine_conv_features=128,
        nb_features=6,
        **kwargs)


def classification_model_airnext_50(**kwargs):
    base_model = AirNext()
    return BaseClassificationModel(
        base_model,
        base_model_features=2048,
        combine_conv_features=128,
        nb_features=6,
        **kwargs)


def classification_model_se_preresnext26b(**kwargs):
    base_model = SE_PreResNet_BC_26b()
    return BaseClassificationModel(
        base_model,
        base_model_features=2048,
        combine_conv_features=128,
        nb_features=6,
        **kwargs)


def classification_model_resnext50(**kwargs):
    base_model = ResNext50()
    return BaseClassificationModel(
        base_model,
        base_model_features=2048,
        nb_features=6,
        **kwargs)


if __name__ == '__main__':
    base_model = pretrainedmodels.resnet34()
    model = ClassificationModelResnetCombineLastVariable(
        base_model,
        base_model_features=512,
        nb_features=6,
        base_model_l1_outputs=64)

    x5 = torch.zeros((4, 5, 384, 384))
    print(model(x5).shape)

    x1 = torch.zeros((4, 1, 384, 384))
    print(model(x1).shape)
