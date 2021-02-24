#!/usr/bin/env python3 

from __future__ import print_function

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image

import tensorflow as tf
import keras2onnx

import sys, os
import trtsamplescommon as trtcommon

class PreprocessINCEPTION(object):
    def __init__(self, yolo_input_resolution):
        self.yolo_input_resolution = yolo_input_resolution

    def process(self, input_image_path):
        image_raw, image_resized = self._load_and_resize(input_image_path)
        image_preprocessed = self._shuffle_and_normalize(image_resized)
        return image_raw, image_preprocessed

    def _load_and_resize(self, input_image_path):
        image_raw = Image.open(input_image_path)
        # model size is h,w, PIL resize needs w,h
        new_resolution = (self.yolo_input_resolution[1], self.yolo_input_resolution[0])
        image_resized = image_raw.resize(new_resolution, resample=Image.BICUBIC)
        image_resized = np.array(image_resized, dtype=np.float32, order='C')
        return image_raw, image_resized

    def _shuffle_and_normalize(self, image):
        image /= 255.0
        image = image[:,:,::-1] # RGB to BGR
        image = np.transpose(image, [2, 0, 1]) # HWC to CHW format
        image = np.expand_dims(image, axis=0)  # CHW to NCHW format
        image = np.array(image, dtype=np.float32, order='C') # row-major order
        return image



TRT_LOGGER = trt.Logger()

def get_engine(engine_file_path):
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        raise ValueErorr("BUILD",engine_file_path,"WITH trtexec!")

def main():

    engine_file_path = 'racecar-inceptionv3-model.engine'

    engine = get_engine(engine_file_path)
    context = engine.create_execution_context()

    with open('racecar-labels.txt') as fp:
        LABELS = [line.strip() for line in fp]

    for input_image_path in ('racecar1.jpg','racecar2.jpg','racecar3.jpg'):
        preprocessor = PreprocessINCEPTION((299,299))
        image_raw, image = preprocessor.process(input_image_path)

        inputs, outputs, bindings, stream = trtcommon.allocate_buffers(engine)
        print('Running inference on image {}...'.format(input_image_path))
        inputs[0].host = image
        trt_outputs = trtcommon.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        x = np.argsort(trt_outputs[0])
        t1,t2,t3 = x[-1],x[-2],x[-3]
        print(input_image_path,t1,trt_outputs[0][t1],LABELS[t1])
        print(input_image_path,t2,trt_outputs[0][t2],LABELS[t2])
        print(input_image_path,t3,trt_outputs[0][t3],LABELS[t3])

if __name__ == '__main__':
    main()
