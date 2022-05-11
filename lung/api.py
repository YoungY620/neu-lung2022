import json
import os
from pprint import pprint
import shutil
import uuid
from io import BytesIO
from operator import itemgetter

import click
import cv2
import jsonschema
import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, make_response, request, send_file
from PIL import Image

from lung.core.analyze import analyze_one, train_all
from lung.core.data import DETECTED_CLASSES, DetectionClass, store_data
from lung.utils import _draw_one_box, auto_increase_filename, auto_increase_filepathname, cache_origin_img

bp = Blueprint('api', __name__)


@bp.route('/analysis', methods=['POST'])
def analysis():
    upload_img = request.files.get('image')
    confidence = request.form.get('confidence', default=0, type=float)
    if upload_img == None:
        return make_response(jsonify({"msg": "a image file is needed."}), 400)
    # TODO 检查文件类型

    tmp_dir = os.path.join(os.path.dirname(__file__), "data/tmp")

    im = Image.open(upload_img)
    analysis_dict = analyze_one(im, confidence)
    np_im = np.array(im)

    # draw_boxes(analysis_dict, np_im, save=True, save_dir=tmp_dir)
    cache_origin_img(analysis_dict, np_im, save_dir=tmp_dir)
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


@bp.route('/processing/data/<path:file>', methods=['GET'])
def show_image_with_label(file):
    if not file is None:
        resource = os.path.join(os.path.dirname(__file__), f'data/{file}')
        if not os.path.exists(resource):
            return make_response(404)
        origin_img = cv2.imread(resource)
        boxes = request.args.get('boxes', default=[], type=json.loads)
        print(boxes)
        for bbx in boxes:
            bx_color = (225, 0, 0) if bbx['type'] == 'vessel' else (0, 0, 225)
            _draw_one_box(origin_img, bbx['box'], label=bbx['label'],
                          label_color=bx_color, save=False, show=False)

        _, encoded_img = cv2.imencode(
            '.jpg', origin_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        img_io = BytesIO(encoded_img)
        img_io.seek(0)
        return send_file(img_io, mimetype='image/jpg')

    return make_response(jsonify({"msg": "error"}), 400)


@bp.route("/importbyzip", methods=['POST'])
def import_data():
    '''
    上传文件. 包含:
    data: 包含所有 *.jpg 图片文件的 *.zip 文件. 不允许有重名文件, 二级文件夹将被忽略
    v_label, b_label, overall_label: 记录所有标注信息的 *.csv 文件. 与数据库表列一致
    drop_before: 是否覆盖之前的所有数据
    '''
    csvtmp_dir = os.path.join(os.path.dirname(
        __file__), f"data/tmp/__csv-{uuid.uuid4()}__")
    ziptmp_dir = os.path.join(os.path.dirname(
        __file__), f'data/tmp/__zip-{uuid.uuid4()}__')
    os.makedirs(csvtmp_dir, exist_ok=True)
    os.makedirs(ziptmp_dir, exist_ok=True)
    try:
        error_response, datafile, labelfiles = _check_attachments(ziptmp_dir)
        drop_before = request.form.get(
            "drop_before", default=False, type=lambda xxx: xxx == "true")
        if error_response:
            return error_response
        store_data(datafile, labelfiles, drop_before, ziptmp_dir)
    except:
        return make_response(500)
    finally:
        if os.path.isdir(csvtmp_dir):
            shutil.rmtree(csvtmp_dir)
        if os.path.isdir(ziptmp_dir):
            shutil.rmtree(ziptmp_dir)

    return make_response(jsonify({"msg": "ok"}), 200)


def _check_attachments(tmp_dir):
    error_response = None
    datafile = request.files.get('data')
    if not datafile:
        error_response = make_response(
            jsonify({"msg": "file 'data' is missing."}), 400)

    indexes = [DetectionClass.Vessel, DetectionClass.Bronchus, "overall"]
    labelfiles = {}
    print(request.files)
    for ind in indexes:
        print(f"saving {ind}")
        labelfiles[ind] = os.path.join(tmp_dir, ind+".csv")

    return error_response, datafile, labelfiles


@bp.route("/train", methods=['POST'])
def train():
    formdata = [request.form.get("from_scratch"),
                request.form.get("cl_epoch"),
                request.form.get("detection_epoch"),
                request.form.get("test_ratio")]
    print(*formdata)
    print(*[type(d) for d in formdata])
    from_scratch = request.form.get(
        "from_scratch", default=True, type=lambda xxx: str(xxx).lower() == 'true')
    cl_epoch = request.form.get("cl_epoch", default=200, type=int)
    detection_epoch = request.form.get("detection_epoch", default=50, type=int)
    test_ratio = request.form.get("test_ratio", default=0.2, type=float)
    print(test_ratio, detection_epoch, from_scratch, cl_epoch)
    train_all(test_ratio, detection_epoch, from_scratch, cl_epoch)
    return make_response(jsonify(msg="ok"), 200)


@bp.cli.command("train-all")
@click.option("--cl")
@click.option("--de")
def train_cli(cl, de):
    cl_epoch = int(cl)
    detection_epoch = int(de)

    train_all(simclr_epoch=cl_epoch, yolo_epoch=detection_epoch)


@bp.route("/fetch", methods=['GET'])
def get_all_data():
    data_dir = os.path.join(os.path.dirname(__file__), "./data")
    files = ["bronchus", "vessel", "overall"]
    res_data = []
    for fname in files:
        file_path = os.path.join(data_dir, fname+".csv")
        ds = pd.read_csv(file_path)
        detection_types = [fname] * len(ds)
        ds.insert(loc=0, column='type', value=detection_types)
        ds = ds.to_dict('records')
        res_data.extend(ds)
    res_data.sort(key=itemgetter('file_name'))
    return make_response(jsonify(data=res_data), 200)


@bp.route("/push", methods=["POST"])
def push_images():
    image_data = request.get_json()
    pprint(image_data)
    try:
        schema = {
            "type": "object",
            "properties": {
                "imgs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "originName": {
                                "type": "string"
                            },
                            "detections": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "enum": ["vessel", "bronchus"]
                                        },
                                        "box": {
                                            "type": "array",
                                            "items": {
                                                "type": "number"
                                            },
                                            "maxItems": 4,
                                            "minItems": 4
                                        },
                                        "a": {
                                            "type": "number",
                                            "maximum": 3,
                                            "minimum": 0
                                        },
                                        "b": {
                                            "type": "number",
                                            "maximum": 3,
                                            "minimum": 0
                                        },
                                        "c": {
                                            "type": "number",
                                            "maximum": 2,
                                            "minimum": 0
                                        },
                                        "d": {
                                            "type": "number",
                                            "maximum": 3,
                                            "minimum": 0
                                        }
                                    },
                                    "required": ["type", "box"]
                                }
                            },
                            "e": {
                                "type": "number",
                                "maximum": 5,
                                "minimum": 0
                            }
                        },
                        "required": ["originName", "detections"]
                    }
                }
            },
            "required": ["imgs"]
        }
        jsonschema.validate(image_data, schema)
    except jsonschema.exceptions.ValidationError as e:
        return make_response(str(e), 400)

    image_data = image_data['imgs']

    # save files
    image_dir = os.path.join(os.path.dirname(__file__), "data/images")
    for imItem in image_data:
        cache_path = os.path.join(
            os.path.dirname(__file__), imItem['originName'])
        cache_dir, imgname = os.path.split(cache_path)
        unique_path, _, unique_file \
            = auto_increase_filepathname(os.path.join(image_dir, imgname))
        shutil.copyfile(cache_path, unique_path)
        imItem['originName'] = unique_file
    # e:
    df = pd.DataFrame(columns=["file_name", "e"])
    for imItem in image_data:
        if 'e' not in imItem.keys():
            continue
        df = pd.concat([df, pd.DataFrame([[imItem['originName'], imItem['e']]],
                                        index=[str(uuid.uuid4()).replace("-", "")], columns=["file_name", "e"])])
    df.to_csv(os.path.join(image_dir, "../overall.csv"), mode='a', header=False)
    # yolo detection
    for imItem in image_data:
        yolo_lb_file = os.path.join(
            image_dir, "../labels/", os.path.splitext(imItem['originName'])[0]+".txt")
        for de in imItem['detections']:
            with open(yolo_lb_file, mode='a') as f:
                w, h = de['box'][2]-de['box'][0], de['box'][3]-de['box'][1]
                f.write(
                    f"{DETECTED_CLASSES.index(de['type'])} \
                        {(de['box'][0]+de['box'][2])/2/w} \
                        {(de['box'][1]+de['box'][3])/2/h} \
                        {(de['box'][2]-de['box'][0])/w} \
                        {(de['box'][3]-de['box'][1])/h}\n")
    # a b c d ratings
    vessel_path = os.path.join(os.path.dirname(__file__), "data/vessel.csv")
    bronchus_path = os.path.join(
        os.path.dirname(__file__), "data/bronchus.csv")
    for im in image_data:
        for de in im['detections']:
            line = [str(uuid.uuid4()).replace("-", ""), im['originName']]
            line = line + [str(x) for x in de['box']]
            if de['type'] == "vessel":
                line.append(str(de['d']) if 'd' in de.keys()
                            and de['d'] else "")
            elif de['type'] == 'bronchus':
                line.append(str(de['a']) if 'a' in de.keys()
                            and de['a'] else "")
                line.append(str(de['b']) if 'b' in de.keys()
                            and de['b'] else "")
                line.append(str(de['c']) if 'c' in de.keys()
                            and de['c'] else "")
            else:
                raise ValueError(f"invalid detection type: {de['type']}")
            with open(vessel_path if de['type'] == 'vessel' else bronchus_path, mode='a') as fobj:
                fobj.write(",".join(line)+'\n')
    return make_response(jsonify(msg="ok"), 200)
