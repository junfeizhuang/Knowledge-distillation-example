import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from model.SubModule import conv_bn_relu
from model.SubModule import BasicBlock, Bottleneck, LambdaLayer

# ResNet 50
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self._weights_init()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out = F.avg_pool2d(out3, out3.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, out1, out2, out3
    
    def _weights_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight)


def ResNet20():
    return ResNet(BasicBlock, [3, 3, 3])


def ResNet32():
    return ResNet(BasicBlock, [5, 5, 5])


def ResNet44():
    return ResNet(BasicBlock, [7, 7, 7])


def ResNet56():
    return ResNet(BasicBlock, [9, 9, 9])


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

def get_attention(fm, eps=1e-5):
    am = torch.pow(torch.abs(fm), 2)
    am = torch.sum(am, dim=1, keepdim=True)
    norm = torch.norm(am, dim=(2,3), keepdim=True)
    am = torch.div(am, norm+eps)

    return am
    
