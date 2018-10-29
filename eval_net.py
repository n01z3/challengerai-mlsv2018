from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import os

import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from utils.data.preprocessor import Preprocessor, VideoTestPreprocessor
from utils.meters import AverageMeter

import models
import errno

def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def collate_batch(batch):
    #print(batch)
    out_batch = torch.stack([b[0] for b in batch], 0)
    out_tags = [b[1] for b in batch]
    return out_batch, out_tags

def get_data(data_dir, ann_file, height, width, batch_size, workers, frames_mode):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])


    test_transformer = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalizer,
    ])

    #open annotation_file
    labels = None
    with open(ann_file) as infile:
        labels = infile.readlines()
    infile.close()


    data_loader = DataLoader(
        VideoTestPreprocessor(data_dir, labels, transform=test_transformer, mode = frames_mode),
        num_workers=workers,
        shuffle=False, pin_memory=True)

    return data_loader


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = False
    cudnn.enabled = True

    batch_size = args.batch_size if args.frames_mode == 'single_frame' else 64
 
    data_loader = \
        get_data(args.data_dir, args.ann_file, args.height,
                 args.width, args.batch_size, args.workers, args.frames_mode)


    model = models.create(args.arch, weigths = args.weights,  gpu = args.gpu, n_classes = 63, features = True)

    if args.gpu:
        model = nn.DataParallel(model).cuda()
    else:
        model = nn.DataParallel(model)
    model.eval()

    topk = [AverageMeter() for i in range(4)]

    print(model)
    #acc = AverageMeter()

    with torch.no_grad():
        for i, (inputs, tags) in enumerate(data_loader):
            if args.gpu:
                inputs = inputs.cuda()

            if inputs.dim() > 4:
                bs, n_frames, c, h, w = inputs.size()
                inputs = inputs.view(-1, c, h, w)
                inputs = torch.split(inputs, batch_size, dim = 0)
                output = torch.stack([model(input) for input in inputs])
                #fuse back
                output = output.view(n_frames, -1)
                if av_mode == 'mean':
                    output = torch.mean(output, 0, keepdim = True)
                elif av_mode == 'max':
                    output = torch.max(output, 0, keepdim = True)
            else:
                output = torch.squeeze(model(input))

            if args.gpu:
                output = output.cpu()
            prec = accuracy(output, tags, 4) 
            for i in range(4):
                topk[i].update(prec[i])

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@2 {top2.val:.3f} ({top2.avg:.3f})\t'
                    'Prec@3 {top3.val:.3f} ({top3.avg:.3f})\t'
                    'Prec@4 {top4.val:.3f} ({top4.avg:.3f})\t'.format(
                    i, len(data_loader),
                    top1=topk[0], top2=topk[1],
                    top3=topk[2], top4=topk[3]))   
        
        print(' * Prec@1 {top1.avg:.3f}'.format(top1=topk[0]) )


def accuracy(outputs, tags, topk=5):
    res = np.zeros(topk)
    if outputs.dim == 1:
        return ch_metric(outputs, tags, topk)
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
    parser = argparse.ArgumentParser(description="Network evaluation")
    # data


    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default = 224)
    parser.add_argument('--width', type=int, default = 224)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--topk', '-t', type = int, default = 5)
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--weights', '-w', type = str, metavar = 'PATH', help='path to the checkpoint')
    parser.add_argument('--ann_file', type=str, metavar='PATH', help = "path to the annotation file")
    parser.add_argument('--data_dir', type=str, metavar='PATH', help = "path to the data folder")
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')    
    parser.add_argument('--averaging_mode', type = str, default = 'mean',
    choices=['mean', 'max']) 
    parser.add_argument('--frames_mode', type = str, default = 'all_frames',
    choices=['all_frames, first_frame', 'random_frames'])

    parser.add_argument('--gpu', action='store_true',
                        help="use gpu")
    

    main(parser.parse_args())
