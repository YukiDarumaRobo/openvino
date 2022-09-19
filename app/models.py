import logging

import numpy as np
import cv2 as cv

from openvino.inference_engine import IECore


logger = logging.getLogger(__name__)

class Model(object):

    def __init__(self, model_path):

        ie_core = IECore()
        net = ie_core.read_network(model_path+'.xml',model_path+'.bin')
        self.exec_net = ie_core.load_network(network=net,
                                             device_name='CPU',
                                             num_requests=0)
        self.input_name = next(iter(net.input_info))
        self.output_name = next(iter(net.outputs))
        self.input_size = net.input_info[self.input_name].input_data.shape
        self.output_size = self.exec_net.requests[0].output_blobs[self.output_name].buffer.shape

    def prepare_frame(self, frame):
        _, _, h, w = self.input_size
        input_frame = cv.resize(frame, (h, w))
        input_frame = input_frame.transpose((2,0,1))
        input_frame = np.expand_dims(input_frame, axis=0)
        logger.debug({'action': 'prepare_frame', 
                      'input_size': (h,w), 
                      'input_frame.shape': input_frame.shape})
        return input_frame

    def infer(self, data):
        input_data = {self.input_name: data}
        infer_result = self.exec_net.infer(input_data)[self.output_name]
        logger.debug({'action': 'infer', 
                      'input_data.shape': input_data[self.input_name].shape, 
                      'infer_result.shape': infer_result.shape})
        return infer_result

class FaceDetector(Model):

    def __init__(self, model_path):
        super(FaceDetector, self).__init__(model_path=model_path)
    
    def prepare_data(self, input, frame, confidence=0.5):
        data_array = None
        for data in np.squeeze(input):
            conf = float(data[2])
            xmin = int(data[3] * frame.shape[1])
            ymin = int(data[4] * frame.shape[0])
            xmax = int(data[5] * frame.shape[1])
            ymax = int(data[6] * frame.shape[0])
            if conf > confidence:
                if xmin < 0:
                    xmin = 0
                if ymin < 0:
                    ymin = 0
                if xmax > frame.shape[1]:
                    xmax = frame.shape[1]
                if ymax > frame.shape[0]:
                    ymax = frame.shape[0]
                area = (xmax-xmin)*(ymax-ymin)
                data = [area,xmin,ymin,xmax,ymax]
                try:
                    data_array = np.vstack((data_array,np.array(data)))
                except ValueError:
                    data_array = np.array([data])
        try:
            if len(data_array) > 1:
                data_array = data_array[np.argsort(data_array[:,0])]
        except TypeError:
            pass
        logger.debug({'action': 'prepare_data',
                      'input.shape': input.shape, 
                      'data_array': data_array})
        return data_array


class LandmarksDetector(Model):

    def __init__(self, model_path):
        super(LandmarksDetector, self).__init__(model_path=model_path)
    
    def prepare_frame(self, frame):
        h, w = 48,48
        input_frame = cv.resize(frame, (h,w))
        input_frame = input_frame.transpose((2,0,1))
        input_frame = np.expand_dims(input_frame, axis=0)
        logger.debug({'action': 'prepare_frame', 
                      'input_size': (h,w), 
                      'input_frame.shape': input_frame.shape})
        return input_frame
    
    def prepare_data(self, input):
        data_array = np.squeeze(input)
        return data_array
