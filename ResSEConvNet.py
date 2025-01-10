import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x, drop_prob: float = 0., training: bool = False):

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)  # depthwise conv
        self.act = nn.GELU()
        self.bn = nn.BatchNorm2d(dim)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv(x)  
        x = self.bn(x)
        x = self.act(x)
        
        x = shortcut + self.drop_path(x)
        return x

class DWConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__() 
        self.conv1 = nn.Conv2d(c1, c1, k, s, p, groups=c1)  
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.GELU() 
    def forward(self, x):
        return self.bn(self.act(self.conv1(x))) 

class DownsampleLayers(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.ch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(c1, c1 // 2, 1, 1, 0), 
            nn.ReLU(),
            nn.Conv2d(c1 // 2, c1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.ch2 = nn.Sequential(
            DWConv(c1, c1, k=3),
            nn.Conv2d(c1, int(1.5*(c2-c1)), 1, 1, 1),
            nn.GELU(),
            nn.BatchNorm2d(int(1.5*(c2-c1)))
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, int(1.5*(c2-c1)), 1, 1, 1),
            nn.BatchNorm2d(int(1.5*(c2-c1)))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(int(1.5*(c2-c1)),c2, 1, 1, 1), 
            nn.GELU(),
            nn.BatchNorm2d(c2)
        )
        self.conv3 = nn.Sequential( 
            nn.Conv2d(c2, c2, 3, 2, 1), 
            nn.GELU(),
            nn.BatchNorm2d(c2)
        )

    def forward(self, x):
        a = self.ch1(x)
        b = self.ch2(x)
        x = self.conv1(x * a + x)
        x = self.conv3(self.conv2(x + b))
        return x

class get_model(nn.Module):
    def __init__(self, in_chans: int = 3, num_classes: int = 1000, depths: list = None,
                 dims: list = None, drop_path_rate: float = 0., layer_scale_init_value: float = 1e-6,
                 head_init_scale: float = 1.):
        super().__init__()
        self.downsample_layers = nn.ModuleList() 
        stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=1),  
                  LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = DownsampleLayers(dims[i], dims[i+1]) 
            self.downsample_layers.append(downsample_layer)
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_rate=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x

def model_architecture(num_classes: int):
    model = get_model(depths=[1, 1, 3, 1], 
                     dims= [72, 144, 288,576], 
                     num_classes=num_classes)
    return model




