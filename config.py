import logging
from os import path
from sys import stdout

from flask import Flask

from app.camera import Camera
from app import models


logging.basicConfig(level=logging.DEBUG, 
                    stream=stdout)

WEB_ADDRESS = '0.0.0.0'
WEB_PORT = 5000
PROJECT_ROOT = path.dirname(path.abspath(__file__))
TEMPLATES = path.join(PROJECT_ROOT, 'app/templates')
STATIC_FOLDER = path.join(PROJECT_ROOT, 'app/static')
DEBUG = False

CAMERA_PORT = 0
SAVE_PATH = path.join(PROJECT_ROOT, 'img/out')

FACE_PATH = path.join(PROJECT_ROOT, 
                      'app/intel/face-detection-retail-0005/FP16/face-detection-retail-0005')
LANDMARKS_PATH = path.join(PROJECT_ROOT,
                      'app/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009')


app = Flask(__name__,
            template_folder=TEMPLATES,
            static_folder=STATIC_FOLDER)

face = models.FaceDetector(model_path=FACE_PATH)
landmarks = models.LandmarksDetector(model_path=LANDMARKS_PATH)

camera = Camera(port=CAMERA_PORT,
                save_path=SAVE_PATH,
                face_detector=face,
                landmarks_detector=landmarks)

if DEBUG:
    app.debug = DEBUG