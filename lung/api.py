import os
import uuid
from zipfile import ZipFile
import shutil

from flask import Blueprint, request, jsonify, make_response
import tempfile
from PIL import Image
import numpy as np

from lung.core.analysis import analyze_one
from lung.utils import draw_boxes

bp = Blueprint('api', __name__)


@bp.route('/analysis', methods=['POST'])
def analysis():
    upload_img = request.files.get('image')
    confidence = request.form.get('confidence')
    confidence = float(confidence) if confidence else 0
    if upload_img == None: return make_response(jsonify({"msg": "a image file is needed."}), 500)
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


@bp.route("/upload", methods=['POST'])
def upload_data():
    datafile = request.files.get('data')
    if datafile == None: return make_response(jsonify({"msg":"a zip file is required"}), 400)

    ziptmp_dir = os.path.join(os.path.dirname(__file__), 'data/tmp/zip')
    with ZipFile(datafile) as datazip:
        for name in datazip.namelist():
            datazip.extract(name, ziptmp_dir)
    
    data_dir = os.path.join(os.path.dirname(__file__), "data/images")
    label_dir = os.path.join(os.path.dirname(__file__), "data/labels")
    if not os.path.exists(data_dir): os.mkdir(data_dir)
    if not os.path.exists(label_dir): os.mkdir(label_dir)
    
    for filepath, _, filenames in os.walk(ziptmp_dir):
        for filename in filenames:
            if filename.endswith(".jpg"):
                unique_name = f"{uuid.uuid4()}"
                # 若附带标签，一定是在同路径下的同名 *.txt 文件
                ext = os.path.splitext(filename)
                label_filename = ext[0]+'.txt'
                if os.path.exists(os.path.join(filepath, label_filename)):
                    shutil.move(os.path.join(filepath, label_filename), os.path.join(data_dir, unique_name+".txt"))
                shutil.move(os.path.join(filepath, filename), os.path.join(data_dir, unique_name+".jpg"))
    shutil.rmtree(ziptmp_dir)
    
    return make_response(jsonify({"msg": "ok"}),200)



@bp.route("/import", methods=['POST'])
def import_data():
    pass


@bp.route("/train")
def train():
    pass

