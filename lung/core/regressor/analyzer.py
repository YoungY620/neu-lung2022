import os
from typing import List
from typing_extensions import Self

from PIL import Image
import joblib
import torch
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import cv2 as cv
import pywt

from lung.core.regressor.simclr_resnet18 import ResNetSimCLR
from lung.core.regressor.transforms import get_simclr_encoding_transform

class CoreState:
    TRAINING = 0
    TESTING = 1


def get_flatten_rating_feature(model, im, transforms, device):
    im = im.convert('RGB')
    
    # 抽象特征
    tensor_im = transforms(im.copy()).unsqueeze(0).to(device)
    ftr = np.array(model(tensor_im).cpu().detach()).flatten()

    grey_im = np.array(im.copy().convert('L'))

    # # 像素特征
    # ftr = np.append(ftr, np.array(im.copy().resize((200, 200))).flatten(), axis=0)

    # 灰度共存矩阵
    compress_gray = np.digitize(grey_im, np.linspace(0, 255, 64))
    comatrix = graycomatrix(grey_im, np.linspace(10, 20, num=4), 
                    [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],
                    256, symmetric=True, normed=True)
    for prop in {'contrast', 'dissimilarity','homogeneity', 'energy', 'correlation', 'ASM'}:
        temp = graycoprops(comatrix, prop).flatten()
        ftr = np.append(ftr, temp, axis=0)

    # 小波包变换能量系数
    n_level = 3
    re = []  #第n层所有节点的分解系数
    wp = pywt.WaveletPacket2D(data=grey_im, wavelet='db1',mode='symmetric',maxlevel=n_level)
    for p in [n.path for nodes in wp.get_level(n_level, 'freq') for n in nodes]:
        ftr = np.append(ftr, np.array([float(pow(np.linalg.norm(wp[p].data,ord=None),2))]), axis=0)
    
    return ftr


class Analyzer(object):
    _instance = None

    def __new__(cls, *args, **kwargs) -> Self:
        if cls._instance is None:
            cls._instance = super(Analyzer, cls).__new__(cls)
        return cls._instance

    def __init__(self, device=None, train=False) -> None:
        assert device != None
        self.device = device

        self.indexes = ['a', 'b', 'c', 'd', 'e']
        self.regressors = {}
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models')
        for index in self.indexes:
            self.regressors[index] = joblib.load(os.path.join(model_dir, f"vot_reg_{index}.pk"))

        self.encoder = ResNetSimCLR(base_model="resnet18", out_dim=64).to(self.device)
        checkpoint = torch.load(os.path.join(model_dir, 'simclr_encoder.pth.tar'), map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict']
        self.encoder.load_state_dict(state_dict)


    def analyze(self, index: str, img: Image) -> float:
        assert (index in ['a', 'b', 'c', 'd']), f'unexpected rating index: {index}'

        trans = get_simclr_encoding_transform(size=1280 if index == 'e' else 80)         
        ftr = get_flatten_rating_feature(self.encoder, img, trans, self.device)
        pred = self.regressors[index].predict([ftr])

        return pred[0]

    def train():
        pass
