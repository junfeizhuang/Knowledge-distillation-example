import os
import time
import torch
import logging
import random
import numpy as np

from pathlib import Path
from tensorboardX import SummaryWriter

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def create_logger(log_path, log_filename):
    logger_folder = Path(Path().resolve(),log_path)
    if not logger_folder.exists():
        logger_folder.mkdir(exist_ok=True, parents=True)
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # print on console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # write in log file
    fh = logging.FileHandler(Path(log_path,log_filename))
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

def create_tb_writer(tb_filename):
    if not Path(tb_filename).exists():
        Path(tb_filename).mkdir(exist_ok=True, parents=True)
    summary_writer = SummaryWriter(tb_filename)
    return summary_writer

# def calculate_accuracy(outputs, labels):
#     outputs = np.argmax(outputs, axis=1)
#     return np.sum(outputs==labels)/float(labels.size)

def calculate_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_model(model, epoch, optimizer, model_name, save_model_path, is_best=False):
    model_save_folder = Path(Path().resolve(),save_model_path)
    if not model_save_folder.exists():
        model_save_folder.mkdir(exist_ok=True, parents=True)

    state = model.state_dict()
    
    torch.save({
        'epoch': epoch+1,
        'arch': model_name,
        'state_dict': state,
        'optimizer_dict': optimizer.state_dict()
        },str(Path(model_save_folder,model_name + '_' + str(epoch) +'.pth')))
    
    if is_best:
        torch.save(state, str(Path(model_save_folder,model_name+'_model_best.pth')))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def time_now():
    return time.strftime('%Y-%m-%d_%H:%M:%S_',time.localtime(time.time()))

def restore_from(model, optimizer, ckpt_path, cfg):
    ckpt = torch.load(ckpt_path, map_location = lambda storage, loc: storage.cuda(cfg.gpu_id))
    epoch = ckpt['epoch']
    arch = ckpt['arch']
    ckpt_model_dict = ckpt['state_dict']
    model.load_state_dict(ckpt_model_dict)

    optimizer.load_state_dict(ckpt['optimizer_dict'])
    return model, optimizer, epoch-1, arch

def set_seed(seed=12345):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False