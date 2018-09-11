from __future__ import absolute_import
import os.path as osp

from PIL import Image
import torch


class Preprocessor(object):
    def __init__(self, data_dir, labels, transform=None):
        super(Preprocessor, self).__init__()
        self.data_dir = data_dir
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        line = self.labels[index]
        sp_line = line.split(",")
        fname, tag = sp_line[0], int(sp_line[1])
        fpath = osp.join(self.data_dir, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, tag