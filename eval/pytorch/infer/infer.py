# -*- coding: utf-8 -*-

import random
from .seresnet import se_resnet50
from torch import nn
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import av
from av import time_base as AV_TIME_BASE

class ServerApi(object):
    """
    统一算法预测接口类：
    **注**：
        1.handle为举办方验证接口，该接口必须返回预测分类list eg:[1, 3, 4]，参赛队伍需具体实现该接口
        2.模型装载操作必须在初始化方法中进行
        3.初始化方法必须提供gpu_id参数
        3.其他接口都为参考，可以选择实现或删除
    """
    def __init__(self, gpu_id=0):
        self.transform = None
        self.device = torch.device("cuda:{}".format(gpu_id))
        self.model = self.load_model()
        
        self.model.eval()

    def video_frames(self, file_dir):
        """
        视频截帧
        :param file_dir: 视频路径
        :return:
        """
        return None

    def load_model(self):
        """
        模型装载
        :param gpu_id: 装载GPU编号
        :return:
        """
        model = se_resnet50("infer/weights/checkpoint.pth.tar", gpu = True, n_classes = 63)
        model = model.to(self.device)

        self._init_transformer()
        return model
    
    def _init_transformer(self):
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])


        test_transformer = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalizer,
        ])       

        self.transform = test_transformer 

    def predict(self, file_dir):
        """
        模型预测
        :param file_dir: 预测文件路径
        :return:
        """
        return None
    
    def _get_single_frame(self, cap):
        ret, frame = cap.read()

        img = Image.fromarray(frame)
        if self.transform is not None:
            img = self.transform(img)
        
        return img
    
    def _get_single_item(self, index, cap, video_stream, frames):
        frame = self._get_single_frame(index, cap, video_stream, frames)
        img = frame.to_image()
        if self.transform is not None:
            img = self.transform(img)
        
        return img
    
    def _get_single_frame(self, index, cap, video_stream, frames):
        cap.seek(int(int((index * AV_TIME_BASE) / video_stream.average_rate)), 'frame')
        got_frame = False

        for packet in cap.demux(video_stream):
            if not got_frame:
                for frame in packet.decode():
                    if frame is not None:
                        return frame
                        

    def handle(self, video_dir):
        """
        算法处理
        :param video_dir: 待处理单视频路径
        :return: 返回预测分类列表 eg:[1, 3, 4]
        """
        #print('VIDEO')
        #print(video_dir)
        #open container
        cap = av.open(video_dir, mode = 'r')
        video_stream = next(s for s in cap.streams if s.type == 'video')
        num_frames = video_stream.frames - 30
        #fix
        if num_frames < 0:
            num_frames = int(video_stream.frames)


        t = np.random.choice(num_frames, size=5)
        t = np.sort(t)
        #frames = torch.stack([self._get_single_item(idx, cap, video_stream, num_frames) for idx in t])
        #frames = frames.to(self.device)
        frames = torch.unsqueeze(self._get_single_item(0, cap, video_stream, num_frames), 0).to(self.device)
        #print(frame.shape)
        #close cap
        #cap.release()
        #print('MODEL   ')
        #print(self.model)
        print('FORWARD')
        output = torch.squeeze(self.model(frames))
        _, pred = output.topk(1)
        pred = pred.cpu().numpy()[0]

        return [pred]


