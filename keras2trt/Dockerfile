FROM nvcr.io/nvidia/tensorrt:20.01-py3

# TENSORFLOW SPECIFICALLY BUILT FOR CUDA 10.2
# https://github.com/tensorflow/tensorflow/issues/38194 
# https://github.com/alexshvid/tensorflow/releases/tag/v2.3.1-cuda10.2
WORKDIR /workspace/tmp
RUN wget https://github.com/alexshvid/tensorflow/releases/download/v2.3.1-cuda10.2/tensorflow-2.3.1-cp36-cp36m-linux_x86_64.whl
RUN pip3 install tensorflow-2.3.1-cp36-cp36m-linux_x86_64.whl
RUN pip3 install tensorflow-addons==0.11.2

RUN pip3 install keras2onnx

WORKDIR /workspace

CMD ["/bin/bash"]
