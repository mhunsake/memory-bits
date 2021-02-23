#!/usr/bin/env python3 

import os

import tensorflow as tf
import keras2onnx

def main():

    onnx_file_path =   'imagenet-inceptionv3-model.onnx'

    if not os.path.exists(onnx_file_path):
        # convert pre-trained tensorflow model to onnx
        tf.keras.backend.set_image_data_format('channels_first')
        model = tf.keras.applications.InceptionV3(weights='imagenet')
        onnx_model = keras2onnx.convert_keras(model, model.name)
        keras2onnx.save_model(onnx_model, onnx_file_path)

if __name__ == '__main__':
    main()
