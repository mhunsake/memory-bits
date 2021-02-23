# keras -> onnx -> trt

## inception_v3 pretrained model

modified /opt/tensorrt/samples/python/yolov3_onnx/ to test with inception_v3

1. Build Docker

   nvidia-docker build --network=host -t memory.bits:v1 -f Dockerfile .

2. Launch Docker, mapping in this code

   nvidia-docker run --net=host -it \
        -v `pwd`:/workspace/memory.bits \
        -w /workspace/memory.bits \
   	memory.bits:v1

3. download pre-trained model and convert to onnx

   ./1-pretrained-keras-to-onnx.py


4. onnx->trt and run inference

   ./2-pretrained-onnx-to-trt.py
