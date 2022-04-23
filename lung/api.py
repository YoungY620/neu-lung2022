import os

from flask import Blueprint, request, jsonify, make_response
import tempfile
from PIL import Image, ImageDraw, ImageFont
import uuid

from lung.core.analysis import analyze_one

bp = Blueprint('api', __name__)


@bp.route('/analysis', methods=['POST'])
def analysis():
    upload_img = request.files.get('image')
    # confidence = request.

    tmp_dir = os.path.join(os.path.dirname(__file__), "data/tmp")

    with tempfile.TemporaryFile(mode='w+b', suffix='.jpg') as fobj:
        origin_name = f"{uuid.uuid4()}-origin.jpg"
        upload_img.save(fobj)
        im = Image.open(fobj)
        analysis_dict = analyze_one(im, 0)
        

    return make_response(jsonify(analysis_dict), 200)


@bp.route("/upload", methods=['POST'])
def upload_data():
    pass


@bp.route("/import")
def import_data():
    pass


@bp.route("/train")
def train():
    pass

