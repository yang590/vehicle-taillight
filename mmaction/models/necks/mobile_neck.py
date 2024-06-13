import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, normal_init, xavier_init

from ..builder import NECKS, build_loss


@NECKS.register_module()
class MobileNeck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 loss_weight=0.5,
                 loss_cls=dict(type='CrossEntropyLoss')):
        super().__init__()

        self.loss_weight = loss_weight
        self.loss_cls = build_loss(loss_cls)

        self.classifiers = nn.ModuleList([nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_channels[i], out_channels),
        ) for i in range(len(in_channels))])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.01)

    def forward(self, x, target):
        losses = dict()

        if target.shape == torch.Size([]) or target.ndim == 1:
            target = target.unsqueeze(0)

        losses['loss_aux'] = 0.
        for i in range(len(x)):
            x_i = self.classifiers[i](x[i])
            losses['loss_aux'] += self.loss_weight * self.loss_cls(x_i, target)

        return losses