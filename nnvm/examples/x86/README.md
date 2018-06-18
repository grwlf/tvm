# TVM on X86 Example

This application demonstrates running a ResNet18 using NNVM by a simple C++ application

## Prerequisites

1. A GNU/Linux environment
2. NNVM, TVM compiled with LLVM, and their corresponding Python modules
4. `pip install --user mxnet pillow`

## Running the example

    bash run_example.sh

If everything goes well, you should see a lot of build messages and below them
the text `It's a tabby!`.

## High-level overview

Running this example performs the following steps:

1. Downloads a pre-trained MXNet ResNet and a
   [test image](https://github.com/BVLC/caffe/blob/master/examples/images/cat.jpg)
2. Converts the ResNet to an NNVM graph + library
3. Links the graph JSON definition, params, and runtime library into into an C++ application
4. Runs an executable performing the inference on the image which invokes TVM module.

For more information on building, please refer to the `Makefile`.
