from torch import nn
from torch.nn import functional as F


class ContextualAttention(nn.Module):
    """
    Input should be of shape (N, num_branches, num_features, h, w).
    """

    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.conv = nn.Conv2d(num_features, num_features, 1)

    def forward(self, x):
        batch_size, num_branches, num_features, h, w = x.shape
        a = x.view(batch_size * num_branches, num_features, h, w)
        a = self.conv(a)

        # compute normalized softmax over branches
        a = a.view(batch_size, num_branches, num_features, h, w)
        a = F.softmax(a, dim=1)
        a = a / a.max(dim=1, keepdim=True)[0]

        x = x * a
        return x


class SpatialAttention(nn.Module):
    """
    Input should be of shape (N, num_branches, num_features, h, w).
    """

    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.conv = nn.Conv2d(num_features, num_features, 1)

    def forward(self, x):
        batch_size, num_branches, num_features, h, w = x.shape
        a = x.view(batch_size * num_branches, num_features, h, w)
        a = self.conv(a)

        # compute normalized softmax over HxW
        a = a.view(batch_size * num_branches, num_features, h * w)
        a = F.softmax(a, dim=2)
        a = a / a.max(dim=2, keepdim=True)[0]

        a = a.view(batch_size, num_branches, num_features, h, w)
        x = x * a
        return x
