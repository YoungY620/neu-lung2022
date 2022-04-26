from collections import Counter
from threading import Thread
import os
import shutil
from tkinter import Image
import uuid
from typing import Dict
from zipfile import ZipFile

import pandas as pd
from lung import db
from lung.utils import auto_increase_filepath
import numpy as np


DETECTED_CLASSES = ["vessel", "bronchus"]

class DetectionClass:
    Vessel = "vessel"
    Bronchus = "bronchus"


def store_data(datafile, labelfiles, drop_before, ziptmp_dir=None):
    '''labelfiles: dict, 目标检测类别名及 "overall" 作为键, 标签 *.csv 文件路径为值'''
    if ziptmp_dir == None:
        ziptmp_dir = os.path.join(os.path.dirname(__file__), f'data/tmp/__zip-{uuid.uuid4()}__')
    renamed = _extract_zip_data(datafile, drop_before, ziptmp_dir)
    dfs = _save_csv_to_sql(labelfiles, renamed, drop_before)
    
    # for yolo dataset:
    detect_classes = DETECTED_CLASSES
    for i, cls in enumerate(detect_classes):
        _arrange_yolo_labels(dfs[cls], i)


def _save_csv_to_sql(labelfiles: Dict, renamed, drop_before):
    label_dfs = {}
    for k, v in labelfiles.items():
        # print(k)
        # print(renamed)
        df = pd.read_csv(v)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        for before, after in renamed.items():
            df = df.replace(to_replace={'file_name': before}, value=after)
        df.index = df.index.map(lambda _: str(uuid.uuid4()).replace("-",""))
        # print(df.head())
        df.to_sql(name=f"{k}_annotation", con=db.engine, index=False, \
            if_exists=("replace" if drop_before else "append"))

        csv_path = os.path.join(os.path.dirname(__file__), f"data/{k}.csv")
        df.to_csv(csv_path, mode=("w" if drop_before else "a"))
        label_dfs[k] = df
    return label_dfs

def _extract_zip_data(datafile, drop_before, ziptmp_dir):
    with ZipFile(datafile) as datazip:
        for name in datazip.namelist():
            datazip.extract(name, ziptmp_dir)
    data_dir = os.path.join(os.path.dirname(__file__), "data/images")
    if drop_before and os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    if not os.path.exists(data_dir): 
        os.mkdir(data_dir)
    renamed = {}
    files = next(os.walk(ziptmp_dir))
    for filename in files[2]:
        if os.path.isfile(os.path.join(ziptmp_dir, filename)) and filename.endswith(".jpg"):
            unique_name = auto_increase_filepath(os.path.join(data_dir, filename))
            if unique_name != filename:
                renamed[filename] = unique_name
            # print(filename, unique_name)
            shutil.move(os.path.join(ziptmp_dir, filename), os.path.join(data_dir, unique_name))
    
    return renamed


def _arrange_yolo_labels(df, cls):
    '''write label files, return img path strings'''
    label_dir = os.path.join(os.path.dirname(__file__), f"../data/labels")
    try: shutil.rmtree(label_dir)
    finally: 
        os.makedirs(label_dir, exist_ok=True)

    for _, r in df.iterrows():
        label_name = r['file_name'].replace(".jpg", "")
        label_path = os.path.join(label_dir, label_name)
        with open(label_path, mode='a') as f:
            f.write(f"{cls} {(r['xmin']+r['xmax'])/2} {(r['ymin']+r['ymax'])/2} {(r['xmax']-r['xmin'])} {(r['ymax']-r['ymin'])}\n")
    

def get_rating_data(index, df):
    data_dir = os.path.join(os.path.dirname(__file__), "../data/image")
    df = df.copy()
    np_imgs, ratings = [], []
    assert index in ['a', 'b', 'c', 'd', 'e'] and index in df.columns
    filename = None
    img = None
    for i, row in df.iterrows():
        if img == None or row['file_name'] != filename:
            img = Image.open(os.path.join(data_dir, row['file_name'])).convert("RGB")
            img = np.array(img)
        ratings.append(row[index])
        if index == 'e':
            np_imgs.append(np.array(img, copy=True))
        else: 
            np_img_copy = np.copy(img)[row['ymin']:row['ymax']+1, row['xmin']:row['xmax']+1]
            np_imgs.append(np_img_copy)
    
    return np_imgs, ratings


