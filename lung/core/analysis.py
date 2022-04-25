import os
from typing import Any, Dict

import joblib
import numpy as np
import torch
from flask import current_app
from lung.core.simclr.feature import get_flatten_rating_feature
from lung.core.simclr.simclr_resnet18 import ResNetSimCLR
from lung.core.simclr.transforms import get_simclr_encoding_transform
from PIL import Image
from typing_extensions import Self


def analyze_one(img: Image, confidence=0.5) -> Dict[str, Any]:
    img = img.convert('RGB')

    core_dir = os.path.dirname(os.path.abspath(__file__))
    yolo_repo = os.path.join(core_dir, 'yolov5')
    yolo_model = os.path.join(core_dir, 'models/detector_yolov5.pt')
    detector = torch.hub.load(yolo_repo, 'custom', path=yolo_model, source='local')
    box_df = detector(img).pandas().xyxy[0]
    
    res_dict = {}
    res_dict['bronchus'], res_dict['vessel'], res_dict['a'], \
        res_dict['b'], res_dict['c'], res_dict['d'], \
            res_dict['b_conf'], res_dict['v_conf'] = [], [], [], [], [], [], [], []

    device = torch.device('cuda' if current_app.config['device'] == 'cuda' and torch.cuda.is_available() else 'cpu')
    analyzer = Analyzer(device=device, train=False)

    for i, d in box_df.iterrows():
        if d['confidence'] < confidence: continue
        
        indexes = []
        bbox = [d['xmin'], d['ymin'], d['xmax'], d['ymax']]
        if d['name'] == 'bronchus':
            res_dict['bronchus'].append(bbox)
            res_dict['b_conf'].append(d['confidence'])
            indexes.extend(['a', 'b', 'c'])
        elif d['name'] == 'vessel':
            res_dict['vessel'].append(bbox)
            res_dict['v_conf'].append(d['confidence'])
            indexes.extend(['d'])
        else:
            raise NotImplementedError(f'unexpected detection class name: {d["name"]}')

        box_img = img.crop(bbox)

        for ind in indexes:
            rate = analyzer.analyze(index=ind, img=box_img)
            res_dict[ind].append(rate)

    e_rate = analyzer.analyze(index=ind, img=img)
    res_dict['e'] = e_rate

    return res_dict


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

