import os

import joblib
import numpy as np
import torch
from lung.core.regressor.feature import get_flatten_rating_feature
from lung.core.regressor.simclr_resnet18 import ResNetSimCLR
from lung.core.regressor.transforms import get_simclr_encoding_transform
from PIL import Image
from typing_extensions import Self


class Analyzer(object):
    '''全局单例类. 为了尽可能节省加载模型参数时间'''
    _instance = None

    def __new__(cls, device=None, train=False, *args, **kwargs) -> Self:
        assert device != None

        if cls._instance is None:
            cls._instance = super(Analyzer, cls).__new__(cls)

            # sklearn model for regression
            indexes = ['a', 'b', 'c', 'd', 'e']
            cls._instance.regressors = {}
            model_dir = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), '../models')
            for index in indexes:
                cls._instance.regressors[index] = joblib.load(
                    os.path.join(model_dir, f"vot_reg_{index}.pk"))

            # model for abstract feature extraction
            cls._instance.encoder = ResNetSimCLR(
                out_dim=64).to(device)
            checkpoint = torch.load(os.path.join(
                model_dir, 'simclr_encoder.pth.tar'), map_location=device)
            cls._instance.encoder.load_state_dict(checkpoint['state_dict'])

            # for color segmentation
            cls._instance.background_hist = np.load(
                os.path.join(model_dir, 'background.npy'))
            cls._instance.cytoplasm_hist = np.load(
                os.path.join(model_dir, 'cytoplasm.npy'))
            cls._instance.nucleus_hist = np.load(
                os.path.join(model_dir, 'nucleus.npy'))

        cls._instance.device = device

        return cls._instance

    def to(self, device):
        self.device = device
        self.encoder.to(device)

    def analyze(self, index: str, img: Image) -> float:
        assert (index in ['a', 'b', 'c', 'd', 'e']), \
            f'unexpected rating index: {index}'

        trans = get_simclr_encoding_transform(size=1280 if index == 'e' else 80)
        ftr = get_flatten_rating_feature(
            img, index, trans, self.encoder, self.device, \
                self.nucleus_hist, self.cytoplasm_hist, self.background_hist)
        pred = self.regressors[index].predict([ftr])

        return pred[0]

    def train():
        pass
