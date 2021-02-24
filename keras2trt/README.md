# keras -> onnx -> trt

## enviornment

- Build Docker

  nvidia-docker build --network=host -t memory.bits:v1 -f Dockerfile .

- Launch Docker, mapping in this code

  nvidia-docker run --net=host -it \
        -v `pwd`:/workspace/memory.bits \
        -w /workspace/memory.bits \
   	memory.bits:v2


- until update Dockerfile.. add tensorflow-onnx

  pip install git+https://github.com/onnx/tensorflow-onnx
  

## inception_v3 pretrained model

modified /opt/tensorrt/samples/python/yolov3_onnx/ to test with inception_v3


1. download pre-trained model and convert to onnx

   ./1-pretrained-keras-to-onnx.py


2. onnx->trt and run inference

   ./2-pretrained-onnx-to-trt.py


## custom inceptoin_v3 model

PROBLEM WITH DYNAMIC INPUT SHAPE 

In python, can use same methodoloyg as above (keras2onnx, us python to create/serialize engine)
However, this onnx CANNOT (easily) be used with TensorRT.... working with NVIDIA...

3. (keras) saved_model -> onnx -> trtexec

   ./3-tensorflow-to-onnx-fixed-batchsize

4. run inference from serialized engine  

   ./4-inference-from-trt.py


