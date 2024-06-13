'''MobilenetV2 in PyTorch.

See the paper "MobileNetV2: Inverted Residuals and Linear Bottlenecks" for more details.
'''
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from torch.autograd import Variable
from collections import OrderedDict
from ..builder import BACKBONES
from mmaction.utils import get_root_logger


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class Extractor(nn.Module):
    def __init__(
            self, d_model, n_head, qdim, t_size, attn_mask=None,
            mlp_factor=4.0, dropout=0.0, drop_path=0.0
        ):
        super().__init__()

        self.d_model = d_model
        self.t_size = t_size

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ln_3 = nn.LayerNorm(d_model)
        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        nn.init.constant_(self.attn.out_proj.weight, 0.)
        nn.init.constant_(self.attn.out_proj.bias, 0.)

        self.ln_0 = nn.LayerNorm(qdim)
        self.reduction = nn.Linear(qdim, d_model)
        nn.init.xavier_uniform_(self.reduction.weight)

        self.ln_1 = nn.LayerNorm(d_model)
        d_mlp = round(mlp_factor * d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_mlp)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(dropout)),
            ("c_proj", nn.Linear(d_mlp, d_model))
        ]))
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.constant_(self.mlp[-1].weight, 0.)
        nn.init.constant_(self.mlp[-1].bias, 0.)

        self.attn_mask = attn_mask

    def attention(self, x, y):

        d_model = self.ln_1.weight.size(0)
        q = (x @ self.attn.in_proj_weight[:d_model].T) + self.attn.in_proj_bias[:d_model]

        k = (y @ self.attn.in_proj_weight[d_model:-d_model].T) + self.attn.in_proj_bias[d_model:-d_model]
        v = (y @ self.attn.in_proj_weight[-d_model:].T) + self.attn.in_proj_bias[-d_model:]
        Tx, Ty, N = q.size(0), k.size(0), q.size(1)
        q = q.view(Tx, N, self.attn.num_heads, self.attn.head_dim).permute(1, 2, 0, 3)
        k = k.view(Ty, N, self.attn.num_heads, self.attn.head_dim).permute(1, 2, 0, 3)
        v = v.view(Ty, N, self.attn.num_heads, self.attn.head_dim).permute(1, 2, 0, 3)
        aff = (q @ k.transpose(-2, -1) / (self.attn.head_dim ** 0.5))

        aff = aff.softmax(dim=-1)
        out = aff @ v
        out = out.permute(2, 0, 1, 3).flatten(2)
        out = self.attn.out_proj(out)
        return out

    def forward(self, x, y):
        x = self.reduction(self.ln_0(x))

        x = x + self.drop_path(self.attention(self.ln_2(x), self.ln_3(y)))

        x = x + self.drop_path(self.mlp(self.ln_1(x)))
        return x



def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1, 1, 1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == (1, 1, 1) and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

@BACKBONES.register_module()
class MobileNetV2Custom(nn.Module):
    def __init__(self, pretrained=None, sample_size=224, width_mult=1.):
        super(MobileNetV2Custom, self).__init__()
        self.pretrained = pretrained

        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        # interverted_residual_setting = [
        #     # t, c, n, s
        #     [1, 16, 1, (1, 1, 1)],
        #     [6, 24, 2, (2, 2, 2)],
        #     [6, 32, 3, (2, 2, 2)],
        #     [6, 64, 4, (2, 2, 2)],
        #     [6, 96, 3, (1, 1, 1)],
        #     [6, 160, 3, (2, 2, 2)],
        #     [6, 320, 1, (1, 1, 1)],
        # ]

        #custom
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, (1, 1, 1)],#1
            [6, 24, 2, (1, 2, 2)],#3
            [6, 32, 3, (1, 2, 2)],#6
            [6, 64, 4, (1, 2, 2)],#10
            [6, 96, 3, (1, 1, 1)],#13
            [6, 160, 3, (1, 2, 2)],#16
            [6, 320, 1, (1, 1, 1)],#17
        ]

        # building first layer
        assert sample_size % 16 == 0.
        input_channel = int(input_channel * width_mult)
        self.input_channel = input_channel
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel

        # custom start
        self.return_layers = [0, 6, 13, 17, 18]

        self.return_channels = [input_channel, 32, 96, 320, self.last_channel]

        self.dpe = nn.ModuleList([
            nn.Conv3d(self.return_channels[i_layer], self.return_channels[i_layer],
                      kernel_size=3, stride=1, padding=1, bias=True, groups=self.return_channels[i_layer])
            for i_layer in range(1, len(self.return_channels))
        ])

        for m in self.dpe:
            nn.init.constant_(m.bias, 0.)

        dpr = [x.item() for x in torch.linspace(0, 0.4, len(self.return_layers)-1)]
        self.dec = nn.ModuleList([
            Extractor(
                self.return_channels[i_layer], n_head=8, qdim=self.return_channels[i_layer-1], t_size=8,
                mlp_factor=4.0, dropout=0.5, drop_path=dpr[i_layer-1],
            ) for i_layer in range(1, len(self.return_channels))
        ])

        # custom end

        # self.features = [conv_bn(3, input_channel, (1, 2, 2))]
        self.features = [conv_bn(3, input_channel, (2, 2, 2))]  #custom
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else (1, 1, 1)
                self.features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        # self.features = nn.Sequential(*self.features)
        self.features = nn.ModuleList(self.features)   #custom

        # building classifier
        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(self.last_channel, num_classes),
        # )

        # self.classifiers = nn.ModuleList([nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(self.return_channels[i_layer], num_classes),
        # ) for i_layer in range(1, len(self.return_channels))])


        # self._initialize_weights()

    def forward(self, x):
        # x = self.features(x)
        # x = F.avg_pool3d(x, x.data.size()[-3:])
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        # return x

        outs = []
        j = -1
        cls_token = torch.zeros(1, x.shape[0], self.input_channel)
        for i, layer in enumerate(self.features):
            x = layer(x.contiguous())
            if i in self.return_layers:

                if i == 0:
                    cls_token = x.permute(2, 3, 4, 0, 1).flatten(0, 2).mean(0, keepdim=True)
                else:
                    j += 1
                    tmp_x = x.clone()
                    tmp_x = tmp_x + self.dpe[j](x.clone())
                    tmp_x = tmp_x.permute(2, 3, 4, 0, 1).flatten(0, 2)
                    cls_token = self.dec[j](cls_token, tmp_x)
                    outs.append(cls_token.squeeze(0))

        return outs


    def init_weights(self, pretrained=None):
        if pretrained:
            self.pretrained = pretrained

        def _init_weights(m):
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        def _load_state_dict():
            checkpoint = torch.load(self.pretrained, map_location='cpu')
            state_dict = self.state_dict()

            common_param = dict()
            for k, v in checkpoint['state_dict'].items():
                if k[7:] in state_dict and v.shape == state_dict[k[7:]].shape:
                    common_param[k[7:]] = v

            state_dict.update(common_param)
            self.load_state_dict(state_dict)

        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            _load_state_dict()

        else:
            raise TypeError('pretrained must be a str or None')


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")


def get_model(**kwargs):
    """
    Returns the model.
    """
    model = MobileNetV2Custom(**kwargs)
    return model



