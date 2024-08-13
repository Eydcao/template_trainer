import torch


def _asinh_ratio(x, rand_scale):
    # NOTE asinh squashes the input to a range of -1 to 1; our dataset has massively different magnitudes.
    return torch.asinh(x / rand_scale)


def asinh_ratio_loss(target, pred, log10_max, log10_min, tc_rng):
    # generate a random float from 0 to 1, on the same device as target, using tc_rng
    rand_float = torch.rand(1, dtype=torch.float32, generator=tc_rng).to(target.device)
    rand_scale = torch.pow(10.0, log10_min + rand_float * (log10_max - log10_min))
    pred_asinh_ratio_mean = torch.mean(
        _asinh_ratio(pred, rand_scale), dim=0, keepdim=True
    )
    SStot = torch.mean(
        (_asinh_ratio(pred, rand_scale) - pred_asinh_ratio_mean) ** 2, dim=0
    )
    SSres = torch.mean(
        (_asinh_ratio(pred, rand_scale) - _asinh_ratio(target, rand_scale)) ** 2, dim=0
    )
    R2 = 1.0 - SSres / SStot
    return torch.sum((1.0 - R2) ** 2)


def r_squared(target, pred):
    return 1 - torch.mean((target - pred) ** 2, dim=0) / torch.mean(
        (target - torch.mean(target, dim=0, keepdim=True)) ** 2, dim=0
    )


def mean_relative_error(target, pred):
    return torch.mean(torch.abs((pred - target) / (target + 1e-6)), dim=0)


def mean_squared_logarithmic_error(target, pred):
    target_positive = torch.abs(target) + 1
    pred_positive = torch.abs(pred) + 1
    return torch.mean(
        (torch.log10(target_positive) - torch.log10(pred_positive)) ** 2, dim=0
    )


def log_10sigma(pred, target):
    target_positive = torch.abs(target) + 1
    pred_positive = torch.abs(pred) + 1
    sigma = torch.sqrt(
        torch.mean(
            (torch.log10(pred_positive) - torch.log10(target_positive)) ** 2, dim=0
        )
        / torch.mean(torch.log10(target_positive) ** 2, dim=0)
    )
    return sigma
