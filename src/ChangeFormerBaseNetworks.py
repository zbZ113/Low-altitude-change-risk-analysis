import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, 
                 activation='prelu', norm=None):
        super().__init__()
        
        layers = [nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)]
        
        # 归一化层
        if norm == 'batch':
            layers.append(nn.BatchNorm2d(output_size))
        elif norm == 'instance':
            layers.append(nn.InstanceNorm2d(output_size))
        
        # 激活函数
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'lrelu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        
        self.block = nn.Sequential(*layers)
        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.block(x)

class DeconvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True,
                 activation='prelu', norm=None):
        super().__init__()
        
        layers = [nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)]
        
        if norm == 'batch':
            layers.append(nn.BatchNorm2d(output_size))
        elif norm == 'instance':
            layers.append(nn.InstanceNorm2d(output_size))
        
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'lrelu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        
        self.block = nn.Sequential(*layers)
        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.block(x)

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, x):
        return self.conv(x)

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=1)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels, scale=0.1):
        super().__init__()
        self.scale = scale
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * self.scale
        out = out + residual
        return out

# 均衡学习率实现
class EqualLR:
    def __init__(self, name):
        self.name = name

    def __call__(self, module, input):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()
        setattr(module, self.name, weight * math.sqrt(2 / fan_in))

    @staticmethod
    def apply(module, name):
        weight = getattr(module, name)
        delattr(module, name)
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        fn = EqualLR(name)
        module.register_forward_pre_hook(fn)
        return fn

def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)
    return module