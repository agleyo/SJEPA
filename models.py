import math

import torch
import torch.nn as nn
from torchvision.models import resnet18


class _SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        x = x.clamp(-10.0, 10.0)
        return grad_output / (1.0 + (math.pi / 2 * x).pow(2))


def surrogate_spike(x):
    return _SurrogateSpike.apply(x)


class LIF(nn.Module):
    def __init__(self, beta=0.9, threshold=0.5,
                 thr_min=0.05, thr_max=1.5,
                 learnable_threshold=False):
        super().__init__()
        self.beta = beta
        self.thr_min = thr_min
        self.thr_max = thr_max
        self.learnable_threshold = learnable_threshold

        t = torch.tensor(float(threshold))
        if learnable_threshold:
            self._threshold = nn.Parameter(t)
        else:
            self.register_buffer("_threshold", t)

    @property
    def threshold(self):
        return self._threshold.clamp(self.thr_min, self.thr_max)

    def init_leaky(self, shape, device):
        return torch.zeros(shape, device=device)

    def forward(self, x, mem):
        mem = self.beta * mem + x
        pre_mem = mem
        spike = surrogate_spike(mem - self.threshold)
        mem = mem * (1.0 - spike.detach())
        return spike, mem, pre_mem


class Encoder(nn.Module):
    def __init__(self, emb_dim=256, spiking=False):
        super().__init__()
        self.spiking = spiking
        self.probe_mem = False
        m = resnet18(weights=None)
        m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        m.maxpool = nn.Identity()
        self.body = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4, m.avgpool,
        )
        self.fc = nn.Linear(512, emb_dim)
        self.fc_norm = nn.LayerNorm(emb_dim)
        if spiking:
            self.lif = LIF(beta=0.9, threshold=0.5, learnable_threshold=True)

    def forward(self, x):
        h = self.fc(self.body(x).flatten(1))
        h = self.fc_norm(h)
        if self.spiking:
            mem = self.lif.init_leaky(h.shape, h.device)
            spike, mem, pre_mem = self.lif(h, mem)
            if self.probe_mem:
                return pre_mem
            return spike
        return h


class Projector(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, spiking=False):
        super().__init__()
        self.spiking = spiking
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )
        self.out_norm = nn.LayerNorm(out_dim)
        if spiking:
            self.lif = LIF(beta=0.9, threshold=0.5, learnable_threshold=True)

    def forward(self, x):
        h = self.net(x)
        h = self.out_norm(h)
        if self.spiking:
            mem = self.lif.init_leaky(h.shape, h.device)
            h, _, _ = self.lif(h, mem)
        return h
