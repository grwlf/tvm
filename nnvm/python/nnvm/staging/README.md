Reverse-engineering TensorFlow model
====================================

This module defines a procedure `stage_tensorflow`.  It takes a
frozen `GraphDef` and produces NNVM DSL source in Python.

Below is a simple usage example. We take `MODEL_PB` which is a TensorFLow model
in Protobuf format and convert it to equivalent NNVM DSL.

```Python

...

from nnvm.staging import stage_tensorflow

with tf.Session(graph=tf.Graph()) as sess:
  with FastGFile(MODEL_PB, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name="")
    graphdef=sess.graph.as_graph_def(add_shapes=True)

sym,params=stage_tensorflow(graphdef,"out.py")
```

File `out.py` should be produced. Its contents may look like the following:


```Python
...
sym_379800752 = _sym.broadcast_mul(sym_74132896,sym_291471616,name="Rcnn_ctcV3/static_batch_normalization_1/batchnorm/mul_2")
sym_105726960 = _sym.broadcast_sub(sym_160781776,sym_379800752,name="Rcnn_ctcV3/static_batch_normalization_1/batchnorm/sub")
sym_85399920 = _sym.broadcast_add(sym_77353008,sym_105726960,name="Rcnn_ctcV3/static_batch_normalization_1/batchnorm/add_1")
sym_85290336 = _sym.Variable(name="Rcnn_ctcV3/initial_conv/conv2d_1/kernel",shape=(11, 31, 1, 32))
sym_74133280 = _sym.Variable(name="Rcnn_ctcV3/initial_conv/conv2d_1/bias",shape=(32,))
sym_125191280 = _sym.pad(sym_85399920,pad_width=((0, 0), (4, 5), (15, 15), (0, 0)))
sym_133685744 = _sym.conv2d(sym_125191280,sym_85290336,dilation=(1, 1),layout="NHWC",strides=(2, 2),padding=[0, 0],kernel_size=(11, 31),channels=32,kernel_layout="HWIO",name="Rcnn_ctcV3/initial_conv/conv2d_1/convolution",use_bias=False)
...
```

This shoudl be a valid Python code which may be used for furhter experiments and optimisations.
