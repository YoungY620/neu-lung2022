import os
import click
from flask import Blueprint
import requests
from lung.core.analyze import train_all

bp = Blueprint('cli', __name__, cli_group='cli')

@bp.cli.command("train-all")
@click.option("--cl")
@click.option("--de")
def train_cli(cl, de):
    cl_epoch = int(cl)
    detection_epoch = int(de)

    train_all(simclr_epoch=cl_epoch, yolo_epoch=detection_epoch)
