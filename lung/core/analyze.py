import os
import random
import shutil
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import torch
from flask import current_app
from lung.core.data import DETECTED_CLASSES, get_rating_data
from lung.core.feature import get_flatten_rating_feature
from lung.core.simclr.cl_data import ContrastiveLearningDataset
from lung.core.simclr.simclr import SimCLR
from lung.core.simclr.simclr_resnet18 import ResNetSimCLR
from lung.core.simclr.transforms import get_simclr_encoding_transform
from lung.core.yolov5 import train as yolo
from PIL import Image
from sklearn.ensemble import AdaBoostRegressor, VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from typing_extensions import Self


def analyze_one(img: Image, confidence=0.5) -> Dict[str, Any]:
    device = torch.device(
        'cuda' if current_app.config['DEVICE'] == 'cuda' and torch.cuda.is_available() else 'cpu')
    img = img.convert('RGB')
    analyzer = ModelGroup(device=device, train=False)

    box_df = analyzer.detector(img).pandas().xyxy[0]

    res_dict = {}
    res_dict['bronchus'], res_dict['vessel'], res_dict['a'], \
        res_dict['b'], res_dict['c'], res_dict['d'], \
        res_dict['b_conf'], res_dict['v_conf'] = [], [], [], [], [], [], [], []

    for _, d in box_df.iterrows():
        if d['confidence'] < confidence:
            continue

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
            raise NotImplementedError(
                f'unexpected detection class name: {d["name"]}')

        box_img = img.crop(bbox)

        for ind in indexes:
            rate = analyzer.analyze(index=ind, img=box_img)
            res_dict[ind].append(rate)

    e_rate = analyzer.analyze(index=ind, img=img)
    res_dict['e'] = e_rate

    return res_dict


def train_all(test_ratio=0.2, yolo_epoch=0, from_scratch=False, simclr_epoch=0):
    device = torch.device(
        'cuda' if current_app.config['DEVICE'] == 'cuda' and torch.cuda.is_available() else 'cpu')
    data_dir = os.path.join(os.path.dirname(__file__), "../data")

    analyzer = ModelGroup(device=device, train=True)

    config_file = os.path.join(os.path.dirname(__file__), "../data/yolo.yml")
    if yolo_epoch != 0: 
        analyzer.train_yolo(config_file, yolo_epoch, from_scratch)
    if simclr_epoch != 0:
        analyzer.train_simclr(from_scratch, simclr_epoch)

    v_csv = os.path.join(data_dir, "vessel.csv")
    b_csv = os.path.join(data_dir, "bronchus.csv")
    o_csv = os.path.join(data_dir, "overall.csv")
    vdf = pd.read_csv(v_csv)
    bdf = pd.read_csv(b_csv)
    odf = pd.read_csv(o_csv)
    indexes = ['a', 'b', 'c', 'd', 'e']
    for ind in indexes:
        if ind in ['a', 'b', 'c']:
            im_x, y = get_rating_data(ind, bdf)
        elif ind in ['d']:
            im_x, y = get_rating_data(ind, vdf)
        else:
            im_x, y = get_rating_data(ind, odf)
        analyzer.train_regressor(ind, im_x, y, test_ratio)


class ModelGroup(object):
    '''全局单例类. 为了尽可能节省加载模型参数时间'''
    _instance = None

    def __new__(cls, device=None, train=False, *args, **kwargs) -> Self:
        assert device != None

        if cls._instance is None:
            cls._instance = super(ModelGroup, cls).__new__(cls)

            # sklearn model for regression
            indexes = ['a', 'b', 'c', 'd', 'e']
            cls._instance.regressors = {}
            model_dir = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), 'models')
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

            # yolo
            core_dir = os.path.dirname(os.path.abspath(__file__))
            yolo_repo = os.path.join(core_dir, 'yolov5')
            yolo_model = os.path.join(core_dir, 'models/detector_yolov5.pt')
            cls._instance.detector = torch.hub.load(
                yolo_repo, 'custom', path=yolo_model, source='local').to(device)

        cls._instance.to(device)

        return cls._instance

    def to(self, device):
        self.device = device
        self.encoder.to(device)

    def analyze(self, index: str, img: Image) -> float:
        assert (index in ['a', 'b', 'c', 'd', 'e']), \
            f'unexpected rating index: {index}'

        trans = get_simclr_encoding_transform(
            size=1280 if index == 'e' else 80)
        ftr = get_flatten_rating_feature(
            img, index, trans, self.encoder, self.device,
            self.nucleus_hist, self.cytoplasm_hist, self.background_hist)
        pred = self.regressors[index].predict([ftr])

        return pred[0]

    def train_simclr(self, from_scratch, simclr_epoch):
        data_dir = os.path.join(os.path.dirname(__file__), "../data")
        cl_dataset = ContrastiveLearningDataset(
            os.path.join(data_dir, "images"))
        train_loader = torch.utils.data.DataLoader(
            cl_dataset, batch_size=10, shuffle=True,
            num_workers=2, pin_memory=True, drop_last=True)
        device = torch.device(
            'cuda' if current_app.config['DEVICE'] == 'cuda' and torch.cuda.is_available() else 'cpu')
        model = ResNetSimCLR(out_dim=64)
        if not from_scratch:
            model.load_state_dict(self.encoder.state_dict())
        optimizer = torch.optim.Adam(
            params=model.parameters(), lr=0.00006, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
        args = {
            'device': device,
            'batch_size': 10,
            'temperature': 0.07,
            'fp16_precision': True,
            'log_every_n_steps': 100,
            'epochs': simclr_epoch,
            'arch': 'resnet18',
            'enable_cuda': current_app.config['DEVICE'] == 'cuda',
            'n_views': 4,
            'resume': False,
            'resume_model_path': './checkpoint_0200.pth.tar',
            'log_dir': './log'
        }
        simclr = SimCLR(model=model, optimizer=optimizer,
                        scheduler=scheduler, **args)
        simclr.train(train_loader)
        self.encoder = model

    def train_yolo(self, cfg_path, yolo_epoch, from_scratch):
        core_dir = os.path.dirname(os.path.abspath(__file__))
        yolo_repo = os.path.join(core_dir, 'yolov5')
        yolo_model = os.path.join(core_dir, 'models/detector_yolov5.pt')
        if from_scratch:
            yolo_model_path = os.path.join(
                os.path.dirname(__file__), "yolov5.pt")
        else:
            yolo_model_path = os.path.join(
                core_dir, 'models/detector_yolov5.pt')
        yolo.run(imgsz=640, batch=16, epochs=yolo_epoch, workers=current_app.config['YOLO_WORKERS'],
                 data=cfg_path, weights=yolo_model_path, exist_ok=True, \
                     device=current_app.config['DEVICE'])
        yolo_pt = os.path.join(os.path.dirname(
            __file__), "yolov5/runs/train/exp/weights/best.pt")
        shutil.copyfile(yolo_pt, os.path.join(
            os.path.dirname(__file__), "models/detector_yolov5.pt"))
        self.detector = torch.hub.load(
            yolo_repo, 'custom', path=yolo_model, source='local').to(self.device)

    def train_regressor(self, index, im_x, y, test_ratio):
        trans = get_simclr_encoding_transform()

        def ftr_converter(input): return get_flatten_rating_feature(
            input, index, trans, self.encoder, self.device,
            self.nucleus_hist, self.cytoplasm_hist, self.background_hist)
        x = np.array([ftr_converter(im) for im in im_x])
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, random_state=42, test_size=test_ratio)

        svr_boost = AdaBoostRegressor(base_estimator=SVR(C=1.0, epsilon=0.2))
        tr_pip = Pipeline([('discretizer', KBinsDiscretizer(n_bins=5, encode="onehot", strategy='uniform')),
                           ('tree', DecisionTreeRegressor(max_depth=5))])
        tree_boost = AdaBoostRegressor(base_estimator=tr_pip)
        estimators = [
            ('svr', svr_boost), ('tree', tree_boost)
        ]
        vot_reg = VotingRegressor(estimators=estimators)
        vot_reg.fit(X_train, y_train)         # 全部特征
        test_s, train_s = vot_reg.score(
            X_test, y_test), vot_reg.score(X_train, y_train)
        print(f"training of '{index}' just completed.")
        print(f"testing score: {test_s}, training score: {train_s}")

        self.regressors[index] = vot_reg
        save_path = os.path.join(os.path.dirname(
            __file__), f"models/vot_reg_{index}.pk")
        joblib.dump(vot_reg, save_path)
