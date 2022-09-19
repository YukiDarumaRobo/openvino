import logging
import math
import statistics

import cv2 as cv


logger = logging.getLogger(__name__)

class Camera(object):

    def __init__(self, port, save_path, face_detector, landmarks_detector):
        self.capture = cv.VideoCapture(port)
        self.path = save_path
        self.face_detector = face_detector
        self.landmarks_detector = landmarks_detector
        logger.debug({'action': '__init__',
                      'port': port,
                      'path': save_path})
    
    def __del__(self):
        self.capture.release()
        logger.debug({'action': '__del__'})
    
    def get_frame(self):
        _, frame = self.capture.read()
        logger.debug({'action': 'get_frame',
                      'frame': frame.shape})
        return frame
    
    def face(self, in_frame, out_frame):
        input_frame = self.face_detector.prepare_frame(in_frame)
        out = self.face_detector.infer(input_frame)
        faces = self.face_detector.prepare_data(out, in_frame)
        try:
            for idx,face in enumerate(faces):
                cv.rectangle(out_frame, (face[1],face[2]), (face[3],face[4]), color=(0,0,255), thickness=3)
                cv.putText(out_frame, str(idx), (face[3],face[4]), cv.FONT_HERSHEY_PLAIN, 2, (0,255,255), 1, cv.LINE_AA)
            logger.debug({'action': 'detect_face', 
                          'faces': faces.shape})
        except TypeError:
            pass
        return out_frame
    
    def landmarks(self, in_frame, out_frame):
        color_picker = [(255,    0,  0),
                        (  0,  255,  0),
                        (  0,    0,255),
                        (  0,  255,255),
                        (255,    0,255),]
        parts = ['right_eye',
                 'left_eye',
                 'nose',
                 'right_mouth',
                 'left_mouth',]
        input_frame = self.face_detector.prepare_frame(in_frame)
        output = self.face_detector.infer(input_frame)
        faces = self.face_detector.prepare_data(output,in_frame)
        try:
            for face in faces:
                face_frame = in_frame[face[2]:face[4],face[1]:face[3]]
                input_frame = self.landmarks_detector.prepare_frame(face_frame)
                output = self.landmarks_detector.infer(input_frame)
                landmarks = self.landmarks_detector.prepare_data(output)
                for i in range(5):
                    x = int(landmarks[2*i] * face_frame.shape[1]) + face[1]
                    y = int(landmarks[2*i+1] * face_frame.shape[0]) + face[2]
                    cv.circle(out_frame, (x, y), 10, color_picker[i], thickness=-1)
                    cv.putText(out_frame, str(i), (x,y-10), cv.FONT_HERSHEY_PLAIN, 2, color_picker[i], 1, cv.LINE_AA)
                    logger.debug({'action': 'detect_landmarks',
                                  'part': parts[i], 'x': x, 'y': y})
        except TypeError:
            pass
        return out_frame
    
    def sunglasses(self, in_frame, out_frame):
        input_frame = self.face_detector.prepare_frame(in_frame)
        output = self.face_detector.infer(input_frame)
        faces = self.face_detector.prepare_data(output,in_frame)
        try:
            for face in faces:
                face_frame = in_frame[face[2]:face[4],face[1]:face[3]]
                input_frame = self.landmarks_detector.prepare_frame(face_frame)
                output = self.landmarks_detector.infer(input_frame)
                landmarks = self.landmarks_detector.prepare_data(output)
                x, y = [], []
                for i in range(2):
                    x.append(int(landmarks[2*i] * face_frame.shape[1]) + face[1])
                    y.append(int(landmarks[2*i+1] * face_frame.shape[0]) + face[2])
                x_center = statistics.mean(x)
                y_center = statistics.mean(y)
                L = int(math.sqrt(face[0]))
                cv.rectangle(out_frame,
                              (x_center-L//5,y_center-L//50),
                              (x_center+L//5,y_center+L//50),
                              (0,0,0),
                              thickness=-1)
                for i in range(2):
                    cv.circle(out_frame,(x[i],y[i]),int(L/(2*math.pi)),(0,0,0),thickness=-1)
                logger.debug({'action': 'sunglasses',
                              '(x,y)':(x,y), '(x_center,y_center)':(x_center,y_center)})
        except TypeError:
            pass
        return out_frame