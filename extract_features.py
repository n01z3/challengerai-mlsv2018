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
from utils.data.preprocessor import Preprocessor
from utils.extra_func import mkdir_if_missing

import models
import errno

def get_data(data_dir, ann_file, height, width, batch_size, workers):

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
        Preprocessor(data_dir, labels, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    return data_loader


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = False
    cudnn.enabled = True
 
    data_loader = \
        get_data(args.data_dir, args.ann_file, args.height,
                 args.width, args.batch_size, args.workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.create(args.arch, stop_layer = 'fc')

    model = nn.DataParallel(model).to(device)
    #model = nn.DataParallel(model)
    model.eval()

    mkdir_if_missing(args.out_dir)
    print(model)

    with torch.no_grad():
        for i, (input, fname, tag) in enumerate(data_loader):
            #print(torch.squeeze(output))
            input = input.to(device)
            output = torch.squeeze(model(input))
            tag = torch.unsqueeze(tag.float(), dim = 1)
            features = torch.cat((tag, output.cpu().float()), dim = 1)
            
            if i % 1000 == 0:
                print('[{}/{}]'.format(i, len(data_loader)))

            torch.save(features, osp.join(args.out_dir, 'torch_features_{}.th'.format(i)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Features extractor")
    # data


    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default = 224)
    parser.add_argument('--width', type=int, default = 224)
    parser.add_argument('--seed', type=int, default=1)
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--weigths', '-w', type = str, metavar = 'PATH', help='path to the checkpoint')
    parser.add_argument('--ann_file', type=str, metavar='PATH', help = "path to the annotation file")
    parser.add_argument('--data_dir', type=str, metavar='PATH', help = "path to the data folder")
    parser.add_argument('--out_dir', type = str, metavar='PATH', help = "path to the output folder")
    parser.add_argument('--n_frames', type = int, default = 4)

    main(parser.parse_args())
