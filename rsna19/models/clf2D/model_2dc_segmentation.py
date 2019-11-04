import torch
import torch.nn as nn
import torch.nn.functional as F

import pretrainedmodels


def avg_max_pool_2d(x):
    avg_pool = F.avg_pool2d(x, x.shape[2:])
    max_pool = F.max_pool2d(x, x.shape[2:])
    x_avg_max_pool = torch.cat((avg_pool, max_pool), 1)
    return x_avg_max_pool.view(x_avg_max_pool.size(0), -1)


class SeparableConv(nn.Module):
    def __init__(self, nin, nout, **kwargs):
        super().__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin, **kwargs)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, level):
        super().__init__()
        self.level = level
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.block(x)


class DecoderBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, level):
        super().__init__()
        self.level = level
        self.block = nn.Sequential(
            SeparableConv(in_channels, out_channels),
            nn.ReLU(),
            SeparableConv(out_channels, out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.block(x)


class DecoderBlockBN(nn.Module):
    def __init__(self, in_channels, out_channels, level):
        super().__init__()
        self.level = level
        self.block = nn.Sequential(
            SeparableConv(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            SeparableConv(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.block(x)


def init_zero_non_center(combine_conv, center_slice):
    nn.init.zeros_(combine_conv.bias.data)
    with torch.no_grad():
        combine_conv.weight.data[:, :, :center_slice, :, :] = 0
        combine_conv.weight.data[:, :, center_slice + 1:, :, :] = 0


class ClassificationModelResnetCombineLastVariable(nn.Module):
    def __init__(self,
                 base_model,
                 DecoderBlock,
                 base_model_features,
                 base_model_l1_outputs,
                 nb_features,
                 filters=16,
                 nb_input_slices=5,
                 dropout=0):
        super().__init__()
        self.dropout = dropout
        self.base_model = base_model
        self.nb_features = nb_features
        self.nb_input_slices = nb_input_slices
        self.base_model_features = base_model_features
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.l1 = nn.Conv2d(1, base_model_l1_outputs, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_model_l1_outputs)

        self.combine_conv4 = nn.Conv3d(512, 512, kernel_size=(nb_input_slices, 1, 1))
        self.combine_conv3 = nn.Conv3d(256, 256, kernel_size=(nb_input_slices, 1, 1))
        self.combine_conv2 = nn.Conv3d(128, 128, kernel_size=(nb_input_slices, 1, 1))
        self.combine_conv1 = nn.Conv3d(64, 64, kernel_size=(nb_input_slices, 1, 1))

        center_slice = (nb_input_slices - 1) // 2
        init_zero_non_center(self.combine_conv4, center_slice=center_slice)
        init_zero_non_center(self.combine_conv3, center_slice=center_slice)
        init_zero_non_center(self.combine_conv2, center_slice=center_slice)
        init_zero_non_center(self.combine_conv1, center_slice=center_slice)

        # self.fc1 = nn.Linear(512 * 2 + nb_features, 128)
        # self.fc2 = nn.Linear(128, nb_features)

        self.fc = nn.Linear(512 * 2 + nb_features, nb_features)

        self.center = DecoderBlock(512, filters * 16, level=5)
        self.dec5 = DecoderBlock(512 + filters * 16, filters * 16, level=4)
        self.dec4 = DecoderBlock(256 + filters * 16, filters * 8, level=3)
        self.dec3 = DecoderBlock(128 + filters * 8, filters * 8, level=2)
        self.dec2 = DecoderBlock(64 + filters * 8, filters * 4, level=1)

        nb_segmentation_features = 7

        self.fc_segmentation = nn.Conv2d(filters * 4, nb_segmentation_features, kernel_size=1)

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        self.base_model.requires_grad = True
        for param in self.base_model.parameters():
            param.requires_grad = True
        self.l1.requires_grad = True
        self.bn1.requires_grad = True

    def combine_slices(self, x, conv, batch_size, nb_input_slices):
        base_model_features = x.shape[1]
        x = x.view(batch_size, nb_input_slices, base_model_features, x.shape[2], x.shape[3])  # BxSxCxHxW

        x = x.permute((0, 2, 1, 3, 4))  # BxCxSxHxW

        if nb_input_slices == self.nb_input_slices:
            x = conv(x)  # BxCx1xHxW
        else:
            slice_offset = (self.nb_input_slices - nb_input_slices) // 2
            x = F.conv3d(x,
                         conv.weight[:, :, slice_offset:slice_offset+nb_input_slices, :, :],
                         conv.bias)

        x = x.view(batch_size, x.shape[1], x.shape[3], x.shape[4])
        return x

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        nb_input_slices = inputs.shape[1]

        x = inputs
        x = x.view(batch_size*nb_input_slices, 1, inputs.shape[2], inputs.shape[3])
        x = self.l1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x0 = self.maxpool(x)

        x1 = self.base_model.layer1(x0)
        x2 = self.base_model.layer2(x1)
        x3 = self.base_model.layer3(x2)
        x4 = self.base_model.layer4(x3)

        x1_combined = self.combine_slices(x1, self.combine_conv1, batch_size=batch_size,
                                          nb_input_slices=nb_input_slices)
        x2_combined = self.combine_slices(x2, self.combine_conv2, batch_size=batch_size,
                                          nb_input_slices=nb_input_slices)
        x3_combined = self.combine_slices(x3, self.combine_conv3, batch_size=batch_size,
                                          nb_input_slices=nb_input_slices)
        x4_combined = self.combine_slices(x4, self.combine_conv4, batch_size=batch_size,
                                          nb_input_slices=nb_input_slices)

        center = self.center(self.maxpool(x4_combined))
        dec5 = self.dec5(torch.cat([center, x4_combined], 1))
        dec4 = self.dec4(torch.cat([dec5, x3_combined], 1))
        dec3 = self.dec3(torch.cat([dec4, x2_combined], 1))
        dec2 = self.dec2(torch.cat([dec3, x1_combined], 1))
        dec1 = F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=False)

        segmentation_result = torch.sigmoid(self.fc_segmentation(dec1))

        x = avg_max_pool_2d(x4_combined)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, self.training)

        segmentation_sum = torch.sum(segmentation_result, dim=(2, 3))
        x = torch.cat([x, segmentation_sum], dim=1)
        # x = self.fc1(x)
        # x = torch.relu(x)
        # cls = self.fc2(x)

        cls = self.fc(x)

        return cls, segmentation_result


class ClassificationModelResnetCombineLastVariable2(nn.Module):
    def __init__(self,
                 base_model,
                 DecoderBlock,
                 base_model_l1_outputs,
                 nb_features,
                 filters=16,
                 nb_input_slices=5,
                 dropout=0):
        super().__init__()
        self.dropout = dropout
        self.base_model = base_model
        self.nb_features = nb_features
        self.nb_input_slices = nb_input_slices
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.l1 = nn.Conv2d(1, base_model_l1_outputs, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_model_l1_outputs)

        self.combine_conv4 = nn.Conv3d(512, 256, kernel_size=(nb_input_slices, 1, 1))
        self.combine_conv3 = nn.Conv3d(256, 128, kernel_size=(nb_input_slices, 1, 1))
        self.combine_conv2 = nn.Conv3d(128, 64, kernel_size=(nb_input_slices, 1, 1))
        self.combine_conv1 = nn.Conv3d(64, 64, kernel_size=(nb_input_slices, 1, 1))

        center_slice = (nb_input_slices - 1) // 2
        init_zero_non_center(self.combine_conv4, center_slice=center_slice)
        init_zero_non_center(self.combine_conv3, center_slice=center_slice)
        init_zero_non_center(self.combine_conv2, center_slice=center_slice)
        init_zero_non_center(self.combine_conv1, center_slice=center_slice)

        # self.fc1 = nn.Linear(512 * 2 + nb_features, 128)
        # self.fc2 = nn.Linear(128, nb_features)
        nb_segmentation_features = 7

        self.fc = nn.Linear(256 * 2 + nb_segmentation_features, nb_features)

        self.dec5 = DecoderBlock(256, filters * 16, level=4)
        self.dec4 = DecoderBlock(128 + filters * 16, filters * 8, level=3)
        self.dec3 = DecoderBlock(64 + filters * 8, filters * 4, level=2)

        self.fc_segmentation = nn.Conv2d(filters * 4, nb_segmentation_features, kernel_size=1)

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        self.base_model.requires_grad = True
        for param in self.base_model.parameters():
            param.requires_grad = True
        self.l1.requires_grad = True
        self.bn1.requires_grad = True

    def combine_slices(self, x, conv, batch_size, nb_input_slices):
        base_model_features = x.shape[1]
        x = x.view(batch_size, nb_input_slices, base_model_features, x.shape[2], x.shape[3])  # BxSxCxHxW

        x = x.permute((0, 2, 1, 3, 4))  # BxCxSxHxW

        if nb_input_slices == self.nb_input_slices:
            x = conv(x)  # BxCx1xHxW
        else:
            slice_offset = (self.nb_input_slices - nb_input_slices) // 2
            x = F.conv3d(x,
                         conv.weight[:, :, slice_offset:slice_offset+nb_input_slices, :, :],
                         conv.bias)

        x = x.view(batch_size, x.shape[1], x.shape[3], x.shape[4])
        return x

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        nb_input_slices = inputs.shape[1]

        x = inputs
        x = x.view(batch_size*nb_input_slices, 1, inputs.shape[2], inputs.shape[3])
        x = self.l1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x0 = self.maxpool(x)

        x1 = self.base_model.layer1(x0)
        x2 = self.base_model.layer2(x1)
        x3 = self.base_model.layer3(x2)
        x4 = self.base_model.layer4(x3)

        x2_combined = self.combine_slices(x2, self.combine_conv2, batch_size=batch_size,
                                          nb_input_slices=nb_input_slices)
        x3_combined = self.combine_slices(x3, self.combine_conv3, batch_size=batch_size,
                                          nb_input_slices=nb_input_slices)
        x4_combined = self.combine_slices(x4, self.combine_conv4, batch_size=batch_size,
                                          nb_input_slices=nb_input_slices)

        dec5 = self.dec5(x4_combined)
        dec4 = self.dec4(torch.cat([dec5, x3_combined], 1))
        dec3 = self.dec3(torch.cat([dec4, x2_combined], 1))

        segmentation_result = torch.sigmoid(self.fc_segmentation(dec3))

        x = avg_max_pool_2d(x4_combined)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, self.training)

        segmentation_sum = torch.sum(segmentation_result, dim=(2, 3))
        x = torch.cat([x, segmentation_sum], dim=1)
        # x = self.fc1(x)
        # x = torch.relu(x)
        # cls = self.fc2(x)

        cls = self.fc(x)

        return cls, segmentation_result


class ClassificationModelResnetCombineLastVariable3(nn.Module):
    def __init__(self,
                 base_model,
                 DecoderBlock,
                 base_model_l1_outputs,
                 nb_features,
                 filters=16,
                 nb_input_slices=5,
                 dropout=0):
        super().__init__()
        self.dropout = dropout
        self.base_model = base_model
        self.nb_features = nb_features
        self.nb_input_slices = nb_input_slices
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.l1 = nn.Conv2d(1, base_model_l1_outputs, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_model_l1_outputs)

        self.combine_conv4 = nn.Conv3d(512, 256, kernel_size=(nb_input_slices, 1, 1))
        self.combine_conv3 = nn.Conv3d(256, 128, kernel_size=(nb_input_slices, 1, 1))
        self.combine_conv2 = nn.Conv3d(128, 64, kernel_size=(nb_input_slices, 1, 1))

        center_slice = (nb_input_slices - 1) // 2
        init_zero_non_center(self.combine_conv4, center_slice=center_slice)
        init_zero_non_center(self.combine_conv3, center_slice=center_slice)
        init_zero_non_center(self.combine_conv2, center_slice=center_slice)

        nb_segmentation_features = 7

        self.fc1 = nn.Linear(256 * 2 + nb_segmentation_features, 256)
        self.fc2 = nn.Linear(256, nb_features)

        self.dec5 = DecoderBlock(256, filters * 16, level=4)
        self.dec4 = DecoderBlock(128 + filters * 16, filters * 8, level=3)
        self.dec3 = DecoderBlock(64 + filters * 8, filters * 4, level=2)
        self.fc_segmentation = nn.Conv2d(filters * 4, nb_segmentation_features, kernel_size=1)

        self.output_segmentation = True

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        self.base_model.requires_grad = True
        for param in self.base_model.parameters():
            param.requires_grad = True
        self.l1.requires_grad = True
        self.bn1.requires_grad = True

    def combine_slices(self, x, conv, batch_size, nb_input_slices):
        base_model_features = x.shape[1]
        x = x.view(batch_size, nb_input_slices, base_model_features, x.shape[2], x.shape[3])  # BxSxCxHxW

        x = x.permute((0, 2, 1, 3, 4))  # BxCxSxHxW

        if nb_input_slices == self.nb_input_slices:
            x = conv(x)  # BxCx1xHxW
        else:
            slice_offset = (self.nb_input_slices - nb_input_slices) // 2
            x = F.conv3d(x,
                         conv.weight[:, :, slice_offset:slice_offset+nb_input_slices, :, :],
                         conv.bias)

        x = x.view(batch_size, x.shape[1], x.shape[3], x.shape[4])
        return x

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        nb_input_slices = inputs.shape[1]

        x = inputs
        x = x.view(batch_size*nb_input_slices, 1, inputs.shape[2], inputs.shape[3])
        x = self.l1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x0 = self.maxpool(x)

        x1 = self.base_model.layer1(x0)
        x2 = self.base_model.layer2(x1)
        x3 = self.base_model.layer3(x2)
        x4 = self.base_model.layer4(x3)

        x2_combined = self.combine_slices(x2, self.combine_conv2, batch_size=batch_size,
                                          nb_input_slices=nb_input_slices)
        x3_combined = self.combine_slices(x3, self.combine_conv3, batch_size=batch_size,
                                          nb_input_slices=nb_input_slices)
        x4_combined = self.combine_slices(x4, self.combine_conv4, batch_size=batch_size,
                                          nb_input_slices=nb_input_slices)

        dec5 = self.dec5(x4_combined)
        dec4 = self.dec4(torch.cat([dec5, x3_combined], 1))
        dec3 = self.dec3(torch.cat([dec4, x2_combined], 1))

        segmentation_result = torch.sigmoid(self.fc_segmentation(dec3))

        x = avg_max_pool_2d(x4_combined)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, self.training)

        segmentation_sum = torch.sum(segmentation_result, dim=(2, 3))
        x = torch.cat([x, segmentation_sum], dim=1)

        x = self.fc1(x)
        x = torch.relu(x)
        cls = self.fc2(x)

        if self.output_segmentation:
            return cls, segmentation_result
        else:
            return cls


class ResnetWeightedSegmentatation(nn.Module):
    def __init__(self,
                 base_model,
                 DecoderBlock,
                 base_model_l1_outputs,
                 nb_features,
                 filters=16,
                 nb_input_slices=5,
                 dropout=0):
        super().__init__()
        self.dropout = dropout
        self.base_model = base_model
        self.nb_features = nb_features
        self.nb_input_slices = nb_input_slices
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.l1 = nn.Conv2d(1, base_model_l1_outputs, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_model_l1_outputs)

        self.combine_conv4 = nn.Conv3d(512, 256, kernel_size=(nb_input_slices, 1, 1))
        self.combine_conv3 = nn.Conv3d(256, 128, kernel_size=(nb_input_slices, 1, 1))
        self.combine_conv2 = nn.Conv3d(128, 64, kernel_size=(nb_input_slices, 1, 1))

        center_slice = (nb_input_slices - 1) // 2
        init_zero_non_center(self.combine_conv4, center_slice=center_slice)
        init_zero_non_center(self.combine_conv3, center_slice=center_slice)
        init_zero_non_center(self.combine_conv2, center_slice=center_slice)

        nb_segmentation_features = 7

        self.fc1 = nn.Linear(256 + nb_segmentation_features, 256)
        self.fc2 = nn.Linear(256, nb_features)

        self.dec5 = DecoderBlock(256, filters * 16, level=4)
        self.dec4 = DecoderBlock(128 + filters * 16, filters * 8, level=3)
        self.dec3 = DecoderBlock(64 + filters * 8, filters * 4, level=2)
        self.fc_segmentation = nn.Conv2d(filters * 4, nb_segmentation_features, kernel_size=1)

    def freeze_encoder(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        self.base_model.requires_grad = True
        for param in self.base_model.parameters():
            param.requires_grad = True
        self.l1.requires_grad = True
        self.bn1.requires_grad = True

    def combine_slices(self, x, conv, batch_size, nb_input_slices):
        base_model_features = x.shape[1]
        x = x.view(batch_size, nb_input_slices, base_model_features, x.shape[2], x.shape[3])  # BxSxCxHxW

        x = x.permute((0, 2, 1, 3, 4))  # BxCxSxHxW

        if nb_input_slices == self.nb_input_slices:
            x = conv(x)  # BxCx1xHxW
        else:
            slice_offset = (self.nb_input_slices - nb_input_slices) // 2
            x = F.conv3d(x,
                         conv.weight[:, :, slice_offset:slice_offset+nb_input_slices, :, :],
                         conv.bias)

        x = x.view(batch_size, x.shape[1], x.shape[3], x.shape[4])
        return x

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        nb_input_slices = inputs.shape[1]

        x = inputs
        x = x.view(batch_size*nb_input_slices, 1, inputs.shape[2], inputs.shape[3])
        x = self.l1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x0 = self.maxpool(x)

        x1 = self.base_model.layer1(x0)
        x2 = self.base_model.layer2(x1)
        x3 = self.base_model.layer3(x2)
        x4 = self.base_model.layer4(x3)

        x2_combined = self.combine_slices(x2, self.combine_conv2, batch_size=batch_size,
                                          nb_input_slices=nb_input_slices)
        x3_combined = self.combine_slices(x3, self.combine_conv3, batch_size=batch_size,
                                          nb_input_slices=nb_input_slices)
        x4_combined = self.combine_slices(x4, self.combine_conv4, batch_size=batch_size,
                                          nb_input_slices=nb_input_slices)

        dec5 = self.dec5(x4_combined)
        dec4 = self.dec4(torch.cat([dec5, x3_combined], 1))
        dec3 = self.dec3(torch.cat([dec4, x2_combined], 1))

        segmentation_result = torch.sigmoid(self.fc_segmentation(dec3))

        # use segmentation as a weight for pooling
        segmentation_low_res = torch.max_pool2d(segmentation_result, 8)
        segmentation_low_res = torch.max(segmentation_low_res, dim=1, keepdim=True)[0] + \
                               torch.mean(segmentation_low_res, dim=1, keepdim=True)  # BxHxW
        m = torch.exp(2 * segmentation_low_res)
        a = m / torch.sum(m, dim=(2, 3), keepdim=True)

        x = torch.sum(a*x4_combined, dim=(2, 3))

        # x = avg_max_pool_2d(x4_combined)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, self.training)

        segmentation_sum = torch.sum(segmentation_result, dim=(2, 3))
        x = torch.cat([x, segmentation_sum], dim=1)

        x = self.fc1(x)
        x = torch.relu(x)
        cls = self.fc2(x)

        return cls, segmentation_result


def segmentation_model_resnet34_combine_last_var(**kwargs):
    base_model = pretrainedmodels.resnet34()
    return ClassificationModelResnetCombineLastVariable(
        base_model,
        DecoderBlock=DecoderBlock,
        base_model_features=512,
        nb_features=6,
        base_model_l1_outputs=64,
        **kwargs)


def segmentation_model_resnet34_combine_last_var2(**kwargs):
    base_model = pretrainedmodels.resnet34()
    return ClassificationModelResnetCombineLastVariable2(
        base_model,
        DecoderBlock=DecoderBlock,
        nb_features=6,
        base_model_l1_outputs=64,
        **kwargs)


def segmentation_model_resnet34_combine_last_var2_dec2(**kwargs):
    base_model = pretrainedmodels.resnet34()
    return ClassificationModelResnetCombineLastVariable2(
        base_model,
        DecoderBlock=DecoderBlock2,
        nb_features=6,
        base_model_l1_outputs=64,
        **kwargs)


def segmentation_model_resnet18_bn_filters8(**kwargs):
    base_model = pretrainedmodels.resnet18()
    return ClassificationModelResnetCombineLastVariable3(
        base_model,
        DecoderBlock=DecoderBlockBN,
        nb_features=6,
        base_model_l1_outputs=64,
        filters=8,
        **kwargs)


def segmentation_model_resnet18_bn_filters8_masked(**kwargs):
    base_model = pretrainedmodels.resnet18()
    return ResnetWeightedSegmentatation(
        base_model,
        DecoderBlock=DecoderBlockBN,
        nb_features=6,
        base_model_l1_outputs=64,
        filters=8,
        **kwargs)

if __name__ == '__main__':
    import torchsummary

    # base_model = pretrainedmodels.resnet34()
    # model = ClassificationModelResnetCombineLastVariable2(
    #     base_model,
    #     DecoderBlock=DecoderBlock2,
    #     nb_features=6,
    #     base_model_l1_outputs=64)

    # model = segmentation_model_resnet18_bn_filters8()
    model = segmentation_model_resnet18_bn_filters8_masked()

    x5 = torch.zeros((4, 5, 384, 384))
    x5_cls, x5_seg = model(x5)
    print(x5_cls.shape, x5_seg.shape)

    x1 = torch.zeros((4, 1, 384, 384))
    x1_cls, x1_seg = model(x1)
    print(x1_cls.shape, x1_seg.shape)

    torchsummary.summary(model.cuda(), (5, 384, 284))
