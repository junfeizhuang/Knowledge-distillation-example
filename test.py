import math
import torch
from model.SubModule import *
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR,ExponentialLR,CosineAnnealingLR,ReduceLROnPlateau,CyclicLR,CosineAnnealingWarmRestarts
import numpy as np
# import matplotlib.pyplot as plt
from copy import deepcopy

import model as model

def test2(c,d):
	return c+d
def test1(a,b):
	return test2()(a,b)
test1(3,4)
# class Student(nn.Module):
#     def __init__(self):
#         super(Student, self).__init__()
#         self.conv1 = conv_bn_relu(3,64,3,1,1)
#         self.conv2 = conv_bn_relu(64,64,3,1,1)
#         self.conv3 = conv_bn_relu(64,128,3,1,1)
#         self.pool1 = self.pool2 = self.pool3 = nn.MaxPool2d(2,2)
#         self.fc1 = nn.Linear(4*4*128,256)
#         self.fc2 = nn.Linear(256,10)
#         self.criterion = nn.CrossEntropyLoss()

#     def forward(self,x):
#         x = self.pool1(self.conv1(x))
#         x = self.pool2(self.conv2(x))
#         x = self.pool3(self.conv3(x))
#         N, C, H, W = x.shape
#         x = x.view(N, -1)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x,p=0.5,training=self.training)
#         x = self.fc2(x)
#         return x

# s = Student()
# LR = 100
# epoches = 100

# scheduler_name_list = ['step', 'exp', 'cos','reduce','log','cycle','sgdr']

# def obtain_lr_scheduler(scheduler_name, optimizer):
#     if scheduler_name == 'step':
#         scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
#     elif scheduler_name == 'exp':
#         scheduler = ExponentialLR(optimizer,gamma=0.9)
#     elif scheduler_name == 'cycle':
#         scheduler = CyclicLR(optimizer,base_lr=0.01, max_lr=100, step_size_up=20)
#     elif scheduler_name == 'sgdr':
#         scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10,T_mult=2)
#     elif scheduler_name == 'cos':
#         scheduler = CosineAnnealingLR(optimizer,T_max = 20)
#     elif scheduler_name == 'log':
#         scheduler = np.logspace(math.log10(math.pow(10,100)), math.log10(1), 100)
#     elif scheduler_name == 'reduce':
#         scheduler = ReduceLROnPlateau(optimizer, patience=5, mode='min', factor=0.9, threshold=0.0001, min_lr=0)

        
#     else:
#         raise ValueError('{} learning scheduler is non-existent.'.format(scheduler_name))
#     return scheduler

# for idx, name in enumerate(scheduler_name_list):
#     optimizer = optim.SGD(s.parameters(),lr = LR, momentum=0.9)
#     scheduler = obtain_lr_scheduler(name,optimizer)
#     lr_list = []
#     if name == 'reduce':
#         train_loss = np.append(np.ones((20,1)),np.arange(0,1,0.0125)[::-1])
#     for epoch in range(epoches):
#         if not name == 'sgdr':
#             if name == 'reduce':
#                 scheduler.step(train_loss[epoch])
#             elif not name == 'log' and not name == 'sgdr':
#                 scheduler.step()
#             elif name == 'log':
#                 curLR = scheduler[epoch]
#                 for param_group in optimizer.param_groups:
#                     param_group['lr'] = curLR
#             lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
#         else:
#             for i in range(100):
#                 scheduler.step()
#                 lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        
        
#     # print(optimizer.state_dict()['param_groups'][0]['lr'])
#     plt.subplot(len(scheduler_name_list),1, idx+1)
#     plt.plot(list(range(len(lr_list))), lr_list, color='red', label=name)
#     plt.legend()
# plt.savefig('nice.jpg') 

