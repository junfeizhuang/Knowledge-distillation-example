import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.SubModule import conv_bn_relu
from model.SubModule import BasicBlock, Bottleneck

# resnet 50
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        fm1 = self.layer2(out)
        fm2 = self.layer3(fm1)
        fm3 = self.layer4(fm2)
        out = F.avg_pool2d(fm3, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, fm1, fm2, fm3

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

def get_attention(featuremap):
    out = torch.sum(torch.pow(featuremap,2),dim=1,keepdim=True)
    out = torch.norm(out, dim=(2,3), keepdim=True)
    out = torch.div(out, norm+1e-5)
    retrun out
    
