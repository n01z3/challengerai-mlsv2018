from __future__ import absolute_import
import os.path as osp

from PIL import Image
import torch
import numpy as np
import cv2

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

class TrainPreprocessor(object):
    def __init__(self, data_dir, labels, transform=None):
        super(TrainPreprocessor, self).__init__()
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

        return img, tag

class VideoTrainPreprocessor(object):
    def __init__(self, data_dir, labels, num_frames = 1, transform = None):
        super(VideoTrainPreprocessor, self).__init__()
        self.data_dir = data_dir
        self.labels = labels
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, indices):
        #print('indices')
        #print(indices)
        if isinstance(indices, (tuple, list)):
            return [self._get_multi_items(index) for index in indices]
        return self._get_multi_items(indices)

    def _get_single_item(self, index, cap):
        
        #print('read index')
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)

        ret, frame = cap.read()
        frame = Image.fromarray(frame)
        if self.transform is not None:
            frame = self.transform(frame)

        return frame

    def _get_multi_items(self, index):
        line = self.labels[index]

        fname, tag, frames = line.split(",")
        fpath = osp.join(self.data_dir, fname)

        t = np.random.choice(int(frames), size=self.num_frames)

        cap = cv2.VideoCapture(fpath)
        if self.num_frames == 1:
            return self._get_single_item(t, cap), int(tag)

        frames = [self._get_single_item(idx, cap) for idx in t]
        
        
        cap.release()
        #frames = *frames
        return frames, np.repeat(int(tag), self.num_frames)
        