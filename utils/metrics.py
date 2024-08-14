import torch


def r_squared(target, pred):
    return 1 - torch.mean((target - pred) ** 2, dim=0) / torch.mean(
        (target - torch.mean(target, dim=0, keepdim=True)) ** 2, dim=0
    )
