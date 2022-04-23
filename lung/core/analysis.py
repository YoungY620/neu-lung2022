from typing import Any, Dict
import os

from flask import current_app
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor
import numpy as np

from lung.core.regressor.analyzer import Analyzer


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
        bbox = d['xmin'], d['ymin'], d['xmax'], d['ymax']
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
