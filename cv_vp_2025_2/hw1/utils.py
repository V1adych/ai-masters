from torch.optim.lr_scheduler import _LRScheduler
import math
import requests
from urllib.parse import urlencode
import os

from skimage.metrics import mean_squared_error as mse
from tqdm import tqdm

import numpy as np
import torch
from torch import nn


def download_from_yadisk(short_url: str, filename: str, target_dir: str):
    base_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?"

    final_url = base_url + urlencode(dict(public_key=short_url))
    response = requests.get(final_url)
    download_url = response.json()["href"]

    target_file = os.path.join(target_dir, filename)
    with open(target_file, "wb") as f:
        f.write(requests.get(download_url).content)


def test_model(model, device, test_dataloader):
    PSNRs = []
    for inputs, targets in tqdm(test_dataloader):
        inputs = inputs.to(device)
        outputs = model(inputs).detach().cpu().numpy()
        targets = targets.numpy()
        inputs = inputs.detach().cpu().numpy()
        outputs.shape = 3, 720, 1280  # 3, 512, 512
        targets.shape = 3, 720, 1280  # 3, 512, 512
        inputs.shape = 3, 720, 1280  # 3, 512, 512
        outputs = np.transpose(outputs, axes=[1, 2, 0])
        inputs = np.transpose(inputs, axes=[1, 2, 0])
        targets = np.transpose(targets, axes=[1, 2, 0])
        psnr_ = mse(outputs, targets)
        PSNRs.append(psnr_)
    print(f"Mean MSE: {np.mean(PSNRs)}")
    return np.mean(PSNRs)


def get_position_from_periods(iteration, cumulative_period):
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i


class CosineAnnealingRestartLR(_LRScheduler):
    def __init__(self, optimizer, periods, restart_weights=(1,), eta_min=0, last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_min = eta_min
        assert len(self.periods) == len(self.restart_weights), (
            "periods and restart_weights should have the same length."
        )
        self.cumulative_period = [sum(self.periods[0 : i + 1]) for i in range(0, len(self.periods))]
        super(CosineAnnealingRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch, self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]

        return [
            self.eta_min
            + current_weight
            * 0.5
            * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * ((self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]


def get_scheduler(optimizer):
    return CosineAnnealingRestartLR(
        optimizer,
        periods=[10] * 20,
        restart_weights=[
            1,
            0.5,
            0.5,
            0.5,
            0.5,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
        ],
        eta_min=1e-7,
    )


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
