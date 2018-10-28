# -*- coding: utf-8 -*-

import random
from .seresnet import se_resnet50
from torch import nn
import torch
import cv2
from PIL import Image
import torchvision.transforms as T

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

    def handle(self, video_dir):
        """
        算法处理
        :param video_dir: 待处理单视频路径
        :return: 返回预测分类列表 eg:[1, 3, 4]
        """
        #print('VIDEO')
        #print(video_dir)
        #open cap
        cap = cv2.VideoCapture(video_dir)
        frame = torch.unsqueeze(self._get_single_frame(cap), 0).to(self.device)
        #print(frame.shape)
        #close cap
        cap.release()
        #print('MODEL   ')
        #print(self.model)
        #print('FORWARD')
        output = torch.squeeze(self.model(frame))
        _, pred = output.topk(1)
        pred = pred.cpu().numpy()[0]

        return [pred]


