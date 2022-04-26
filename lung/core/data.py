from collections import Counter
from threading import Thread
import os
import shutil
from PIL import Image
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
        ziptmp_dir = os.path.join(os.path.dirname(
            __file__), f'data/tmp/__zip-{uuid.uuid4()}__')
    renamed = _extract_zip_data(datafile, drop_before, ziptmp_dir)
    dfs = _save_csv(labelfiles, renamed, drop_before)

    # for yolo dataset:
    _arrange_yolo_dataset(drop_before, dfs)
    _write_yolo_dataset_config()


def _arrange_yolo_dataset(drop_before, dfs):
    label_dir = os.path.join(os.path.dirname(__file__), f"../data/labels")
    image_dir = os.path.join(os.path.dirname(__file__), f"../data/images")
    if drop_before and os.path.isdir(label_dir):
        shutil.rmtree(label_dir)
    os.makedirs(label_dir, exist_ok=True)
    detect_classes = DETECTED_CLASSES
    for i, d_cls in enumerate(detect_classes):
        print(i, d_cls)
        for _, r in dfs[d_cls].iterrows():
            h, w, _ = np.array(Image.open(
                os.path.join(image_dir, r['file_name']))).shape
            label_name = r['file_name'].replace(".jpg", ".txt")
            label_path = os.path.join(label_dir, label_name)
            with open(label_path, mode='a') as f:
                f.write(
                    f"{i} {(r['xmin']+r['xmax'])/2/w} {(r['ymin']+r['ymax'])/2/h} {(r['xmax']-r['xmin'])/w} {(r['ymax']-r['ymin'])/h}\n")


def _save_csv(labelfiles: Dict, renamed, drop_before):
    label_dfs = {}
    for k, v in labelfiles.items():
        # print(k)
        # print(renamed)
        df = pd.read_csv(v)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        for before, after in renamed.items():
            df = df.replace(to_replace={'file_name': before}, value=after)
        df.index = df.index.map(lambda _: str(uuid.uuid4()).replace("-", ""))
        # print(df.head())
        # df.to_sql(name=f"{k}_annotation", con=db.engine, index=False, \
        #     if_exists=("replace" if drop_before else "append"))

        csv_path = os.path.join(os.path.dirname(__file__), f"../data/{k}.csv")
        df.to_csv(csv_path, mode=("w" if drop_before else "a"))
        label_dfs[k] = df
    return label_dfs


def _extract_zip_data(datafile, drop_before, ziptmp_dir):
    with ZipFile(datafile) as datazip:
        for name in datazip.namelist():
            datazip.extract(name, ziptmp_dir)
    data_dir = os.path.join(os.path.dirname(__file__), "../data/images")
    if drop_before and os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    renamed = {}
    files = next(os.walk(ziptmp_dir))
    for filename in files[2]:
        if os.path.isfile(os.path.join(ziptmp_dir, filename)) and filename.endswith(".jpg"):
            unique_name = auto_increase_filepath(
                os.path.join(data_dir, filename))
            if unique_name != filename:
                renamed[filename] = unique_name
            # print(filename, unique_name)
            shutil.move(os.path.join(ziptmp_dir, filename),
                        os.path.join(data_dir, unique_name))

    return renamed


def _write_yolo_dataset_config():
    cfg_file = os.path.join(os.path.dirname(__file__), "../data/yolo.yml")
    train_data = './'
    test_data = ""
    cls_names = DETECTED_CLASSES
    data_path = os.path.join(os.path.dirname(__file__), "../data/images")
    dataset_config = f'''# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: {data_path}  # dataset root dir
train: {train_data}  # train images (relative to 'path') 128 images
val: {train_data}  # val images (relative to 'path') 128 images
# test: {test_data} # test images (optional)

# Classes
nc: {len(cls_names)}  # number of classes
names: {str(cls_names)}  # class names'''

    with open(cfg_file, 'w') as f:
        f.write(dataset_config)


def get_rating_data(index, df):
    data_dir = os.path.join(os.path.dirname(__file__), "../data/image")
    df = df.copy()
    np_imgs, ratings = [], []
    assert index in ['a', 'b', 'c', 'd', 'e'] and index in df.columns
    filename = None
    img = None
    for i, row in df.iterrows():
        if img == None or row['file_name'] != filename:
            img = Image.open(os.path.join(
                data_dir, row['file_name'])).convert("RGB")
            img = np.array(img)
        ratings.append(row[index])
        if index == 'e':
            np_imgs.append(np.array(img, copy=True))
        else:
            np_img_copy = np.copy(
                img)[row['ymin']:row['ymax']+1, row['xmin']:row['xmax']+1]
            np_imgs.append(np_img_copy)

    return np_imgs, ratings
