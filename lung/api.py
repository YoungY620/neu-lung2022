import os
import shutil
import uuid
from typing import Dict
import uuid
from zipfile import ZipFile

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, make_response, request
from PIL import Image

from lung.core.analysis import analyze_one
from lung.utils import auto_increase_filepath, draw_boxes
from lung import db

bp = Blueprint('api', __name__)


@bp.route('/analysis', methods=['POST'])
def analysis():
    upload_img = request.files.get('image')
    confidence = request.form.get('confidence')
    confidence = float(confidence) if confidence else 0
    if upload_img == None: return make_response(jsonify({"msg": "a image file is needed."}), 400)
    # TODO 检查文件类型

    tmp_dir = os.path.join(os.path.dirname(__file__), "data/tmp")

    # with tempfile.TemporaryFile(mode='w+b', suffix='.jpg') as fobj:
    #     upload_img.save(fobj)
    im = Image.open(upload_img)
    analysis_dict = analyze_one(im, confidence)
    np_im = np.array(im)
    
    draw_boxes(analysis_dict, np_im, save=True, save_dir=tmp_dir)    
    return make_response(jsonify(analysis_dict), 200)


# show image
@bp.route('/data/<path:file>', methods=['GET'])
def show_photo(file):
    # TODO 检查文件类型
    if not file is None:
        resource = os.path.join(os.path.dirname(__file__), f'data/{file}')
        if not os.path.exists(resource):
            return make_response(404)
        image_data = open(resource, "rb").read()
        response = make_response(image_data)
        response.headers['Content-Type'] = 'image/jpg'
        return response
    
    return make_response(jsonify({"msg": "error"}), 400)


@bp.route("/import-by-zip", methods=['POST'])
def import_data():
    '''
    上传文件. 包含:
    data: 包含所有 *.jpg 图片文件的 *.zip 文件. 不允许有重名文件, 二级文件夹将被忽略
    v_label, b_label, overall_label: 记录所有标注信息的 *.csv 文件. 与数据库表列一致
    drop_before: 是否覆盖之前的所有数据
    '''
    csvtmp_dir = os.path.join(os.path.dirname(__file__), f"data/tmp/__csv-{uuid.uuid4()}__")
    ziptmp_dir = os.path.join(os.path.dirname(__file__), f'data/tmp/__zip-{uuid.uuid4()}__')
    try:
        error_response, datafile, labelfiles = _check_attachments(csvtmp_dir)
        drop_before = (request.form.get("drop_before").lower() == "true")
        if error_response: return error_response
        renamed = _extract_zip_data(datafile, drop_before, ziptmp_dir)
        _save_csv_to_sql(labelfiles, renamed, drop_before)
    except:
        return make_response(500)
    finally:
        if os.path.isdir(csvtmp_dir): shutil.rmtree(csvtmp_dir)
        if os.path.isdir(ziptmp_dir): shutil.rmtree(ziptmp_dir)
        
    return make_response(jsonify({"msg": "ok"}), 200)


def _check_attachments(tmp_dir):
    error_response = None
    datafile = request.files.get('data')
    indexes = ["vessel", "bronchus", "overall"]
    labelfiles = {}
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
    for ind in indexes:
        path = os.path.join(tmp_dir, f"{ind}-{uuid.uuid4()}.csv")
        request.files.get(ind).save(path)
        labelfiles[ind] = path


    if not datafile: 
        error_response =  make_response(jsonify({"msg": "file 'data' is missing."}), 400)
    error_msg = None

    # whether exists
    for k, v in labelfiles.items():
        if v == None:
            error_msg = f"{error_msg}, {k}" if error_msg else k
    if error_msg: 
        error_response =  make_response(jsonify({"msg": f"following label files are missing: \n{error_msg}"}))
    
    # has correct columns:
    for k, v in labelfiles.items():
        std_clms = ["file_name"]
        if k == "bronchus": 
            std_clms = std_clms + ['xmin', 'ymin', 'xmax', 'ymax', 'a', 'b', 'c']
        elif k == "vessel":
            std_clms = std_clms + ['xmin', 'ymin', 'xmax', 'ymax', 'd']
        else:
            std_clms = std_clms + ['e']

        if 0 != len(pd.DataFrame(columns=std_clms).columns.difference(pd.read_csv(v, nrows=0).columns)):
            this_err = f"{k} require columns of {std_clms}"
            error_msg = f"{error_msg}, {this_err}"
    if error_msg: 
        error_response = make_response(jsonify({"msg": f"following label files has wrong columns: \n{error_msg}"}))
    
    return error_response, datafile, labelfiles


def _save_csv_to_sql(labelfiles: Dict, renamed, drop_before):
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



@bp.route("/train")
def train():
    pass

