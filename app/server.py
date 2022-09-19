import logging
from time import sleep

from cv2 import imencode
import flask

import config


logger = logging.getLogger(__name__)
app = config.app
camera = config.camera
mode = {'face': False,
        'landmarks': False,
        'sunglasses': False}

@app.route('/')
def index():
    return flask.render_template('index.html', mode=mode)

def run():
    global app, camera
    try:
        app.run(host=config.WEB_ADDRESS,
                port=config.WEB_PORT,
                threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        del camera
        del app
        sleep(0.1)

@app.route('/api/command/', methods=['POST'])
def command():
    global mode
    cmd = flask.request.form.get('command')
    mode[cmd] = not mode[cmd]
    logger.info({'action': 'command', 'cmd': cmd})
    return flask.jsonify(status='success'), 200

def gen():
    global mode
    while True:
        try:
            in_frame = camera.get_frame()
            out_frame = in_frame.copy()
            if mode['face']:
                out_frame = camera.face(in_frame.copy(), out_frame)
            if mode['landmarks']:
                out_frame = camera.landmarks(in_frame.copy(), out_frame)
            if mode['sunglasses']:
                out_frame = camera.sunglasses(in_frame.copy(), out_frame)
            _, jpeg = imencode('.jpg', out_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        except NameError:
            pass

@app.route('/video/streaming')
def video_feed():
    return flask.Response(gen(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
