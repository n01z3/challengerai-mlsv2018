# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import random
from .seresnet import se_resnet50, se_resnext50_32x4d
from torch import nn
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import av
from av import time_base as AV_TIME_BASE
import pickle
DOCKER_DEBUG = False

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

        self.logreg = None
        self.catboost = None
        self.svm = None
        self.pca = None
        self.scaler = None
        
        self.model.eval()

        self.load_classifiers()

    def video_frames(self, file_dir):
        """
        视频截帧
        :param file_dir: 视频路径
        :return:
        """
        return None

    def load_classifiers(self):
        if DOCKER_DEBUG:
            print('init classifiers')
        with open("infer/weights/pca_model.pkl", "rb") as handle:
            self.pca = pickle.load(handle)
        with open("infer/weights/svm_scaler.pkl", "rb") as handle:
            self.scaler = pickle.load(handle)

        with open("infer/weights/logreg_model.pkl", "rb") as handle:
            self.logreg = pickle.load(handle)
        with open("infer/weights/catboost_model.pkl", "rb") as handle:
            self.catboost = pickle.load(handle)
        with open("infer/weights/svm_model.pkl", "rb") as handle:
            self.svm = pickle.load(handle)
        
        if DOCKER_DEBUG:
            print("init classifiers. done.")

 
    def load_model(self):
        """
        模型装载
        :param gpu_id: 装载GPU编号
        :return:
        """
        model = se_resnext50_32x4d("infer/weights/se_resnext_checkpoint.pth.tar", n_classes = 63, aggr = 'max', features = True)
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
    
    def _make_lst(self, pred):
        lst = []
        for i in pred:
            lst.append(int(i))
        return lst
    
    def _get_single_frame(self, index, cap, video_stream, frames):
        cap.seek(int(int((index * AV_TIME_BASE) / video_stream.average_rate)), 'frame')
        got_frame = False

        for packet in cap.demux(video_stream):
            if not got_frame:
                for frame in packet.decode():
                    if frame is not None:
                        return frame

    def svm_pred(self, feature):
        x = self.scaler.transform(feature)

        pred = self.svm.predict(x)
        return pred

    def _get_frames(self, cap, video_stream, frames):
        frames = []
        k = 0
        for i, frame in enumeraye(cap.decode(video=0):
            frames.append(frame)
            k+=1
            if k == frames:
                break
        return frames

    def _get_items(self, index, cap, video_stream, frames):
        frames = self._get_frames(cap, video_stream, frames)
        imgs = [self.transform(frame.to_image()) for frame in frames]

        return imgs


    def _voting(self, pred_1, pred_2, pred_3, pred_4):
        def most_common(val1, val2, val3, val4):
            lst = [val1, val2, val3, val4]
            return max(set(lst), key = lst.count)
        
        vote = []
        for i, j, k, t in zip(pred_1, pred_2, pred_3, pred_4):
            vote.append(most_common(i, j, k, t))
        return vote

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


        t = np.random.choice(num_frames, size=6)
        t = np.sort(t)
        frames = torch.stack([self._get_single_item(idx, cap, video_stream, num_frames) for idx in t])
        frames = frames.to(self.device)
     
        if DOCKER_DEBUG:
            print('FORWARD')
        pred, features = self.model(frames)
        if DOCKER_DEBUG:
            print ('fordward.done')

        features = features.cpu().detach().numpy().reshape(1, -1)
        scaled_features = self.pca.transform(features)
        
        if DOCKER_DEBUG:
            print('start prediction')

        #get svm prediction
        svm_pred = self.svm_pred(scaled_features)
        catboost_pred = self.catboost.predict(scaled_features)[0]
        logreg_pred = np.argmax(self.logreg.predict_proba(scaled_features), axis = 1)


        if DOCKER_DEBUG:
            print('svm_predict', svm_pred)
            print('catboost_predictt', catboost_pred)
            print('logreg_predict', logreg_pred)

        svm_pred = self._make_lst(svm_pred)
        logreg_pred = self._make_lst(logreg_pred)
        catboost_pred = self._make_lst(catboost_pred)
        _, net_pred = pred.topk(1)
        net_pred = net_pred.cpu().numpy()[0]

        vote = self._voting(svm_pred, logreg_pred, catboost_pred, net_pred)

        ##pred = pred.cpu().numpy()[0]
        if DOCKER_DEBUG:
            print ('vote predict', vote)
        return vote


