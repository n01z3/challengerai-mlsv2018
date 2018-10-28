from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import os
import random
import shutil
import time
import warnings

import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from utils.data.preprocessor import Preprocessor, TrainPreprocessor, VideoTrainPreprocessor
#from utils.data.sampler import RandomFramesSampler
from utils.logging import Logger
from utils.meters import AverageMeter

import models
import errno


working_dir = osp.dirname(osp.abspath(__file__))

def collate_batch(batch):
    #print(batch)
    out_batch = torch.stack([b[0] for b in batch], 0)
    out_tags = [b[1] for b in batch]
    return out_batch, out_tags[0]

def dump_exp_inf(args):
    #open logger

    f = open(osp.join(working_dir, args.logs_dir, 'exp_info.txt'), 'w')

    f.write('experiment was run with following parameters:\n')
    for key, value in vars(args).items():
        f.write('{} : {} \n'.format(key, value))

    f.close()


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def adjust_learning_rate(optimizer, epoch, default_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = default_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_data(train_data_dir, train_ann_file, val_data_dir, val_ann_file, height, width, batch_size, workers):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])


    test_transformer = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalizer,
    ])

    train_transformer = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])


    #open annotation_file
    val_labels = None
    with open(val_ann_file) as infile:
        val_labels = infile.readlines()
    infile.close()

    train_labels = None
    with open(train_ann_file) as infile:
        train_labels = infile.readlines()
    infile.close()


    train_loader = DataLoader(
        VideoTrainPreprocessor(train_data_dir, train_labels, transform=train_transformer, num_frames = args.n_frames),
        batch_size=batch_size, num_workers=workers, shuffle=True,
        pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        VideoTrainPreprocessor(val_data_dir, val_labels, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return train_loader, val_loader


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    cudnn.enabled = True

    sys.stdout = Logger(osp.join(working_dir, args.logs_dir, 'log.txt'))
    dump_exp_inf(args)
 
    train_loader, val_loader = \
        get_data(args.train_data_dir, args.train_ann_file, 
                args.val_data_dir, args.val_ann_file,
                args.height, args.width, args.batch_size, args.workers)


    model = models.create(args.arch, n_classes = 22)

    if args.gpu is not None:
        model = nn.DataParallel(model).cuda(args.gpu)
        criterion = nn.MultiLabelSoftMarginLoss().cuda(args.gpu)
    else:
        model = nn.DataParallel(model)
        criterion = nn.MultiLabelSoftMarginLoss()

    #model = nn.DataParallel(model)
    
    # define loss function (criterion) and optimizer

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    #mkdir_if_missing(args.out_dir)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    best_prec1 = 0

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.module.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename=osp.join(working_dir, args.logs_dir, 'checkpoint.pth.tar'))
    
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    topk = [AverageMeter() for i in range(4)]

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.float()
        if input.dim() > 4:
            input = input.reshape(input.shape[0] * input.shape[1], input.shape[2], input.shape[3], input.shape[4])
            #target  = target.float()
            target = target.reshape(target.shape[0] * target.shape[1], target.shape[2])
            #target = torch.from_numpy(target).float()

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        if args.gpu is not None:
            output = output.cpu()
            target = target.cpu()
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec = accuracy(output, target, topk=4)
        for k in range(4):
            topk[k].update(prec[k], input.size(0))
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\n'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@2 {top2.val:.3f} ({top2.avg:.3f})\t'
                'Prec@3 {top3.val:.3f} ({top3.avg:.3f})\t'
                'Prec@4 {top4.val:.3f} ({top4.avg:.3f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=topk[0], top2=topk[1],
                top3=topk[2], top4=topk[3]))

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    topk = [AverageMeter() for i in range(4)]

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if input.dim() > 4:
                input = input.reshape(input.shape[0] * input.shape[1], input.shape[2], input.shape[3], input.shape[4])
                #target  = target.float()
                target = target.reshape(target.shape[0] * target.shape[1], target.shape[2])
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            target = target.float()
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            if args.gpu is not None:
                output = output.cpu()
                target = target.cpu()
            prec = accuracy(output, target, topk=4)
            for i in range(4):
                topk[i].update(prec[i], input.size(0))
            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\n'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@2 {top2.val:.3f} ({top2.avg:.3f})\t'
                    'Prec@3 {top3.val:.3f} ({top3.avg:.3f})\t'
                    'Prec@4 {top4.val:.3f} ({top4.avg:.3f})\t'.format(
                    i, len(val_loader), batch_time=batch_time,
                    loss=losses, top1=topk[0], top2=topk[1],
                    top3=topk[2], top4=topk[3]))

        print(' * Prec@1 {top1.avg:.3f} Prec@2 {top2.avg:.3f} Prec@3 {top3.avg:.3f} Prec@4 {top4.avg:.3f}'
            .format(top1=topk[0], top2=topk[1], top3=topk[2], top4=topk[3]))

    return topk[0].avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, osp.join(working_dir, args.logs_dir, 'model_best.pth.tar'))

def accuracy(outputs, tags, topk=5):
    res = np.zeros(topk)
    for i in range(outputs.shape[0]):
        res += ch_metric(outputs[i], tags[i], topk)
    res /= outputs.shape[0]

    return res 

def ch_metric(output, tags, topk):
    y = tags.nonzero().numpy().flatten()
    y = set(y)
    res = np.zeros(topk)
    for i in range(1, topk + 1):
        _, pred = output.topk(i)
        pred = set(pred.numpy())
        res[i - 1] = len(set.intersection(pred, y)) / len(set.union(pred, y))
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="network training")
    # data


    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default = 224)
    parser.add_argument('--width', type=int, default = 224)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_frames', type = int, default = 4)
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--train_ann_file', type=str, metavar='PATH', help = "path to the training annotation file")
    parser.add_argument('--train_data_dir', type=str, metavar='PATH', help = "path to the training data folder")
    parser.add_argument('--val_ann_file', type=str, metavar='PATH', help = "path to the validation annotation file")
    parser.add_argument('--val_data_dir', type=str, metavar='PATH', help = "path to the validation data folder")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.0002,
                        help="learning rate of all parameters")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))

    global args
    args = parser.parse_args()
    
    main()