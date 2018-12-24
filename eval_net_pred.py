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

def get_data(data_dir, ann_file, height, width, batch_size, workers, frames_mode, n_frames, label_mode):

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
        VideoTestPreprocessor(data_dir, labels, transform=test_transformer, mode = frames_mode, 
        num_frames = n_frames, label_mode = label_mode),
        num_workers=workers, batch_size=batch_size,
        shuffle=False, pin_memory=True)

    return data_loader


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = False
    cudnn.enabled = True

    
 
    data_loader = \
        get_data(args.data_dir, args.ann_file, args.height,
                 args.width, args.batch_size, args.workers, 
                 args.frames_mode, args.n_frames, args.label_mode)


    model = models.create(args.arch, weights = args.weights,  gpu = args.gpu, n_classes = 63, aggr = 'max', features = True)

    if args.gpu:
        model = nn.DataParallel(model).cuda()
    else:
        model = nn.DataParallel(model)
    model.eval()

    topk = [AverageMeter() for i in range(args.topk)]

    if args.out_dir is not None:
        mkdir_if_missing(args.out_dir)
    print(model)
    #acc = AverageMeter()

    with torch.no_grad():
        for i, (inputs, tags) in enumerate(data_loader):
            if args.gpu:
                inputs = inputs.cuda()
    
            if inputs.dim() > 4:
                bs, n_frames, c, h, w = inputs.size()
                inputs = inputs.view(-1, c, h, w)
                pred, features = model(inputs, bs, n_frames)
                #fuse back
            else:
                pred, features = torch.squeeze(model(inputs))
            pred = nn.functional.log_softmax(pred)
            if args.gpu:
                pred = pred.cpu()
                features = features.cpu()
                tags = tags.cpu()

            if args.out_dir is not None:
                torch.save(pred, osp.join(args.out_dir, 'torch_pred_{}.th'.format(i)))
                torch.save(tags, osp.join(args.out_dir, 'torch_tags_{}.th'.format(i)))
                torch.save(features, osp.join(args.out_dir, 'torch_features_{}.th'.format(i)))

            prec = accuracy(pred, tags, args.topk) 

            for k in range(args.topk):
                topk[k].update(prec[k])

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@2 {top2.val:.3f} ({top2.avg:.3f})\t'.format(
                    i, len(data_loader),
                    top1=topk[0], top2=topk[1]))   
        
        print(' * Prec@1 {top1.avg:.3f}'.format(top1=topk[0]) )


def accuracy(outputs, tags, topk=5):
    res = np.zeros(topk)
    if outputs.dim() == 1:
        return ch_metric(outputs, tags, topk)
    for i in range(outputs.shape[0]):
        res += ch_metric(outputs[i], tags[i], topk)
    res /= outputs.shape[0]

    return res 

def ch_metric(output, tags, topk):
    tags = tags.numpy()
    y = tags[np.where(tags != -1)]
    y = set(y)
    res = np.zeros(2)
    #print(output)    
    #get prediction for the first tag 
    _, pred = output.topk(1)
    pred = set(pred.numpy())
    res[0] = len(set.intersection(pred, set([tags[0]]))) / float(len(set.union(pred, set([tags[0]]))))
    if len(y) == 1:
        res[1] = res[0]
    else:
        _, pred = output.topk(len(y))
        pred = set(pred.numpy())
        res[1] = len(set.intersection(pred, y)) / float(len(set.union(pred, y)))
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Network evaluation")
    # data


    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default = 224)
    parser.add_argument('--width', type=int, default = 224)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--topk', '-t', type = int, default = 3)
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--weights', '-w', type = str, metavar = 'PATH', help='path to the checkpoint')
    parser.add_argument('--ann_file', type=str, metavar='PATH', help = "path to the annotation file")
    parser.add_argument('--data_dir', type=str, metavar='PATH', help = "path to the data folder")
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')  
    parser.add_argument('--n_frames', type = int, default = 4)  
    parser.add_argument('--averaging_mode', type = str, default = 'mean',
    choices=['mean', 'max']) 
    parser.add_argument('--frames_mode', type = str, default = 'all_frames',
    choices=['all_frames', 'first_frame', 'random_frames'])
    parser.add_argument('--label_mode', type = str, default = 'single-class', choices=['single-class', 'multi-class'])
    parser.add_argument('--gpu', action='store_true',
                        help="use gpu")
    parser.add_argument('--out_dir', type = str, default = None, metavar='PATH', help = "path to the output folder")  

    main(parser.parse_args())
