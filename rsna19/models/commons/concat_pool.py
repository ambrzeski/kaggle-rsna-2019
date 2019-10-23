import torch
import torch.nn.functional as F


def concat_pool(x):
    avg_pool = F.avg_pool2d(x, x.shape[2:])
    max_pool = F.max_pool2d(x, x.shape[2:])
    avg_max_pool = torch.cat((avg_pool, max_pool), 1)
    x = avg_max_pool.view(avg_max_pool.size(0), -1)
    return x
