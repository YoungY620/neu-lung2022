# coding: utf-8
import uuid


from lung import db



class BronchusAnnotation(db.Model):
    __tablename__ = 'bronchus_annotation'

    id = db.Column(db.String(32), primary_key=True, default=lambda: uuid.uuid4().hex)
    file_name = db.Column(db.String(50), nullable=False)
    xmin = db.Column(db.Float, nullable=False)
    ymin = db.Column(db.Float, nullable=False)
    xmax = db.Column(db.Float, nullable=False)
    ymax = db.Column(db.Float, nullable=False)
    a = db.Column(db.Float)
    b = db.Column(db.Float)
    c = db.Column(db.Float)



class OverallAnnotation(db.Model):
    __tablename__ = 'overall_annotation'

    id = db.Column(db.String(32), primary_key=True, default=lambda: uuid.uuid4().hex)
    file_name = db.Column(db.String(50), nullable=False)
    e = db.Column(db.Float)



class VesselAnnotation(db.Model):
    __tablename__ = 'vessel_annotation'

    id = db.Column(db.String(32), primary_key=True, default=lambda: uuid.uuid4().hex)
    file_name = db.Column(db.String(50), nullable=False)
    xmin = db.Column(db.Float, nullable=False)
    ymin = db.Column(db.Float, nullable=False)
    xmax = db.Column(db.Float, nullable=False)
    ymax = db.Column(db.Float, nullable=False)
    d = db.Column(db.Float)
