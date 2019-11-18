import os
import torch
import argparse
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR


from config import config
from utils import *
from model.models import *
from model.loss import *

np.random.seed(12345)
torch.manual_seed(12345)
torch.cuda.manual_seed(12345)

def obtain_lr_scheduler(scheduler_name, optimizer, last_epoch):
    if scheduler_name == 'step':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.8, last_epoch=last_epoch)
    elif scheduler_name == 'exp':
        scheduler = ExponentialLR(optimizer,gamma=0.9,last_epoch=last_epoch)
    elif scheduler_name == 'cos':
        scheduler = CosineAnnealingLR(optimizer,T_max = 50,last_epoch=last_epoch)
    else:
        raise ValueError('{} learning scheduler is non-existent.'.format(scheduler_name))
    return scheduler

def train_dml(dataloader, student1_model, student2_model, epoch, \
                optimizer1, optimizer2, lr_scheduler1, lr_scheduler2, \
                logger,summary_writer, cfg):
    student1_model.train()
    student2_model.train()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    logger.info('--------------Train DML Prossing--------------')
    with torch.cuda.device(cfg.gpu_id):
        student1_model.cuda()
        student2_model.cuda()
        for idx, (data, label) in tqdm(enumerate(dataloader)):
            data, label = data.cuda(), label.cuda()
            data, label = Variable(data), Variable(label)
            student1_output, _, _, _ = student1_model(data)
            student2_output, _, _, _ = student2_model(data)

            loss1 = DML_loss_fn(student1_output, student2_output, label, cfg)
            
        
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()
            losses1.update(loss1.item(), data.size(0))

            student1_output, _, _, _ = student1_model(data)
            student2_output, _, _, _ = student2_model(data)
            loss2 = DML_loss_fn(student2_output, student1_output, label, cfg)
            
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
            losses2.update(loss2.item(), data.size(0))

            step = epoch * len(dataloader) + idx
            summary_writer.add_scalar('train/loss1', loss1.data, step)
            summary_writer.add_scalar('train/loss2', loss2.data, step)
            cur_lr = lr_scheduler1.get_lr()[0]
            if idx % cfg.print_freq == 0:
                logger.info('Epoch: [{0}][{1}/{2}] lr : {lr:.5f} \t Train Loss1:{loss1.avg:.5f}\t Train Loss2:{loss2.avg:.5f}'.format(
                epoch + 1, idx + 1, len(dataloader), lr=cur_lr, loss1=losses1, loss2=losses2))

        lr_scheduler1.step()
        lr_scheduler2.step()
    return student1_model, student2_model, optimizer1, optimizer2, lr_scheduler1, lr_scheduler2

def train_at(dataloader, student_model, teacher_model, epoch, optimizer, lr_scheduler, logger,summary_writer, cfg):
    student_model.train()
    teacher_model.eval()
    losses = AverageMeter()
    logger.info('--------------Train AT Prossing--------------')
    with torch.cuda.device(cfg.gpu_id):
        student_model.cuda()
        teacher_model.cuda()
        for idx, (data, label) in tqdm(enumerate(dataloader)):
            data, label = data.cuda(), label.cuda()
            data, label = Variable(data), Variable(label)
            student_output, s_atmp1, s_atmp2, s_atmp3 = student_model(data)
            teacher_output, t_atmp1, t_atmp2, t_atmp3 = teacher_model(data)
            s_atmp1, s_atmp2, s_atmp3 = map(get_attention,[s_atmp1, s_atmp2, s_atmp3])
            t_atmp1, t_atmp2, t_atmp3 = map(get_attention,[t_atmp1, t_atmp2, t_atmp3])
            loss1 = MSE_loss_fn(s_atmp1, t_atmp1)
            loss2 = MSE_loss_fn(s_atmp2, t_atmp2)
            loss3 = MSE_loss_fn(s_atmp3, t_atmp3)

            loss = KD_loss_fn(student_output, label, teacher_output, cfg)
            loss += loss1 + loss2 + loss3
            losses.update(loss.item(), data.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step = epoch * len(dataloader) + idx
            summary_writer.add_scalar('train/loss', loss.data, step)
            cur_lr = lr_scheduler.get_lr()[0]
            if idx % cfg.print_freq == 0:
                logger.info('Epoch: [{0}][{1}/{2}] lr : {lr:.5f} \t Train Loss:{loss.avg:.5f}'.format(
                epoch + 1, idx + 1, len(dataloader), lr=cur_lr, loss=losses))

        lr_scheduler.step()
    return student_model, optimizer, lr_scheduler


def train_kd(dataloader, student_model, teacher_model, epoch, optimizer, lr_scheduler, logger,summary_writer, cfg):
    student_model.train()
    teacher_model.eval()
    losses = AverageMeter()
    logger.info('--------------Train KD Prossing--------------')
    with torch.cuda.device(cfg.gpu_id):
        student_model.cuda()
        teacher_model.cuda()
        for idx, (data, label) in tqdm(enumerate(dataloader)):
            data, label = data.cuda(), label.cuda()
            data, label = Variable(data), Variable(label)
            student_output, _, _, _ = student_model(data)
            teacher_output, _, _, _ = teacher_model(data)
            loss = KD_loss_fn(student_output, label, teacher_output, cfg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), data.size(0))
            step = epoch * len(dataloader) + idx
            summary_writer.add_scalar('train/loss', loss.data, step)
            cur_lr = lr_scheduler.get_lr()[0]
            if idx % cfg.print_freq == 0:
                logger.info('Epoch: [{0}][{1}/{2}] lr : {lr:.5f} \t Train Loss:{loss.avg:.5f}'.format(
                epoch + 1, idx + 1, len(dataloader), lr=cur_lr, loss=losses))

        lr_scheduler.step()
    return student_model, optimizer, lr_scheduler


def train(dataloader, model, epoch, optimizer, lr_scheduler, logger,summary_writer, cfg):
    model.train()
    losses = AverageMeter()
    logger.info('--------------Train Prossing--------------')
    with torch.cuda.device(cfg.gpu_id):
        model.cuda()
        for idx, (data, label) in tqdm(enumerate(dataloader)):
            data, label = data.cuda(), label.cuda()
            data, label = Variable(data), Variable(label)
            output, _, _, _ = model(data)
            loss = CE_loss_fn(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), data.size(0))
            step = epoch * len(dataloader) + idx
            summary_writer.add_scalar('train/loss', loss.data, step)
            cur_lr = lr_scheduler.get_lr()[0]
            if idx % cfg.print_freq == 0:
                logger.info('Epoch: [{0}][{1}/{2}] lr : {lr:.5f} \t Train Loss:{loss.avg:.5f}'.format(
                epoch + 1, idx + 1, len(dataloader), lr=cur_lr, loss=losses))

        lr_scheduler.step()

    return model, optimizer, lr_scheduler
    

def eval(dataloader, model, epoch, logger,summary_writer, cfg):
    model.eval()
    losses = AverageMeter()
    accuracys = AverageMeter()
    logger.info('--------------Evaluate Prossing--------------')
    with torch.no_grad():
        with torch.cuda.device(cfg.gpu_id):
            
            model.cuda()
            for idx, (data, label) in tqdm(enumerate(dataloader)):
                data, label = data.cuda(), label.cuda()
                data, label = Variable(data), Variable(label)
                output, _, _, _ = model(data)
                loss = CE_loss_fn(output, label)
                accuray = calculate_accuracy(output.data, label)[0]
                accuracys.update(accuray)
                losses.update(loss, data.size(0))
                step = epoch * len(dataloader) + idx
                summary_writer.add_scalar('val/loss', loss.data, step)
                summary_writer.add_scalar('val/accuray', accuracys.avg, step)

                if idx % cfg.print_freq == 0:
                    logger.info('Epoch: [{0}][{1}/{2}] Val Loss:{loss.avg:.5f}\t Accuray:{accuray.avg:.3f}'.format(
                    epoch + 1, idx + 1, len(dataloader), loss=losses, accuray=accuracys))
    logger.info('Epoch: {} final accuray {:.3f}'.format(epoch,accuracys.avg))
    return accuracys.avg

def main(cfg):
    set_seed()
    logger = create_logger(cfg.log_save_path, time_now() + cfg.arch + '.log')
    summary_writer = create_tb_writer(os.path.join(cfg.writer_path, time_now() + cfg.arch))

    train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    trainset = torchvision.datasets.CIFAR10(
                                    root='./data',
                                    train=True,
                                    download=False,
                                    transform=train_transform
                                )
    
    testset = torchvision.datasets.CIFAR10(
                                    root='./data',
                                    train=False,
                                    download=False,
                                    transform=test_transform
                                )
    trainloader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True)

    testloader = DataLoader(testset, batch_size=cfg.batch_size, shuffle=False)

    with torch.cuda.device(cfg.gpu_id):
        if 'student' in cfg.arch:
            model = ResNet20().cuda()
        elif cfg.arch == 'teacher':
            model = ResNet56().cuda()

    accuray_max = 0
    is_best = False

    optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate,
                momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    
    if cfg.use_restore_model:
        model, optimizer, epoch, arch = restore_from(model, optimizer, cfg.restore_pth_path,cfg)
        logger.info('restore from {}'.format(cfg.restore_pth_path))
        setattr(cfg,'start_epoch',epoch)
        lr_scheduler = obtain_lr_scheduler(cfg.scheduler_name,optimizer, epoch)
    else:
        lr_scheduler = obtain_lr_scheduler(cfg.scheduler_name,optimizer, -1)


    for epoch in range(cfg.start_epoch,cfg.end_epoch):

        if 'student_kd' == cfg.arch:
            student_model = model
            teacher_model = ResNet56().cuda()
            ckpt = torch.load(cfg.teacher_path, map_location = lambda storage, loc: storage.cuda(cfg.gpu_id))
            teacher_model.load_state_dict(ckpt)
            model, optimizer, lr_scheduler = train_kd(trainloader,student_model, teacher_model, epoch, \
                                            optimizer, lr_scheduler, logger, summary_writer, cfg)
        elif 'student_at' == cfg.arch:
            student_model = model
            teacher_model = ResNet56().cuda()
            ckpt = torch.load(cfg.teacher_path, map_location = lambda storage, loc: storage.cuda(cfg.gpu_id))
            teacher_model.load_state_dict(ckpt)
            model, optimizer, lr_scheduler = train_at(trainloader,student_model, teacher_model, epoch, \
                                            optimizer, lr_scheduler, logger, summary_writer, cfg)
        elif 'student_dml' == cfg.arch:
            student1_model = model
            student2_model = ResNet20().cuda()
            optimizer1 = optimizer
            optimizer2 = optim.SGD(student2_model.parameters(), lr=cfg.learning_rate,
                momentum=cfg.momentum, weight_decay=cfg.weight_decay)
            lr_scheduler1 = lr_scheduler
            lr_scheduler2 = obtain_lr_scheduler(cfg.scheduler_name,optimizer2, -1)
            student1_model, student2_model, optimizer1, optimizer2, lr_scheduler1, lr_scheduler2 = \
                train_dml(trainloader, student1_model, student2_model, epoch, \
                    optimizer1, optimizer2, lr_scheduler1, lr_scheduler2, \
                    logger,summary_writer, cfg)
            model = student1_model
            optimizer = optimizer1
            lr_scheduler = lr_scheduler1

        elif 'student' == cfg.arch or 'teacher' == cfg.arch:
            model, optimizer, lr_scheduler = train(trainloader,model, epoch, \
                                            optimizer, lr_scheduler, logger,summary_writer, cfg)
        else:
            raise ValueError('{} is not proper'.format(cfg.arch))

        accuracy = eval(testloader, model, epoch, logger,summary_writer, cfg)
        if accuracy > accuray_max: 
            is_best = True 
            accuray_max = accuracy
        else:
            is_best = False
        save_model(model, epoch, optimizer, cfg.arch, cfg.model_save_path,is_best=is_best)
    logger.info('accuray_max: {:.3f}'.format(accuray_max))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model_type', type=str, default='', help='student, teacher, student_kd, student_at, or student_dml')
    parser.add_argument('-gpu','--gpu_id', type=int, default=1, help='models and logs are saved here')
    args = parser.parse_args()
    cfg = config()
    setattr(cfg,'arch',args.model_type)
    setattr(cfg,'gpu_id',args.gpu_id)
    main(cfg)

