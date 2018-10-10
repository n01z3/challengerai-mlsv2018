from __future__ import absolute_import
import os.path as osp

from PIL import Image
import torch
import numpy as np
import cv2

import av
from av import time_base as AV_TIME_BASE

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
        #self.cap = cv2.VideoCapture()
        self.current_idx = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, indices):
        #print('indices')
        #print(indices)
        if isinstance(indices, (tuple, list)):
            items = []
            for index in indices:
                items.append(*(self._get_multi_items(index)))
            print(items)
            return items[0]
            #return [self._get_multi_items(index) for index in indices]
        #print('idx:' , indices)
        return self._get_multi_items(indices)
            
    def _get_single_item(self, index, cap, video_stream, frames):
        frame = self._get_single_frame(index, cap, video_stream, frames)
        img = frame.to_image()
        if self.transform is not None:
            img = self.transform(img)
        
        return img

    def _get_multi_items(self, index):
        #got video index        
        line = self.labels[index]

        fname, tag, frames = line.split(",")
        fpath = osp.join(self.data_dir, fname)

        #open container
        cap = av.open(fpath, mode = 'r')
        #get video stream
        video_stream = next(s for s in cap.streams if s.type == 'video')
        num_frames = int(frames) - 30
        #fix
        if num_frames < 0:
            num_frames = int(frames)
            print('debug')
            print(fname, tag, int(frames))

        if self.num_frames == 1:
            img = self._get_single_item(np.random.randint(num_frames), cap, video_stream, num_frames)
            return img, int(tag)
            #return img, int(tag)
           
        t = np.random.choice(num_frames, size=self.num_frames)
        t = np.sort(t)
        frames = torch.stack([self._get_single_item(idx, cap, video_stream, num_frames) for idx in t])
        cap = None
        return frames, np.repeat(int(tag), self.num_frames)
        
    def _get_single_frame(self, index, cap, video_stream, frames):
        cap.seek(int(int((index * AV_TIME_BASE) / video_stream.average_rate)), 'frame')
        got_frame = False

        for packet in cap.demux(video_stream):
            if not got_frame:
                for frame in packet.decode():
                    if frame is not None:
                        return frame
                        


