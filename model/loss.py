import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def CE_loss_fn(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)

def MSE_loss_fn(outputs, labels):
    return torch.nn.MSELoss()(outputs,labels)

def KD_loss_fn(outputs, labels, teacher_outputs, cfg):
    alpha = cfg.alpha
    T = cfg.temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1), \
                                F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)+ \
                                F.cross_entropy(outputs, labels) * (1. - alpha)
    return KD_loss

def DML_loss_fn(outputs1, outputs2, labels, cfg):
    dml_lamdba = cfg.dml_lamdba
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs1, dim=1), \
                                F.softmax(outputs2, dim=1)) * (dml_lamdba)+ \
                                F.cross_entropy(outputs1, labels)
    return KD_loss