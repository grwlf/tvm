import numpy as np
import tvm
from tvm.contrib import graph_runtime
import topi.testing
import nnvm.symbol as sym
import nnvm.compiler
from nnvm.testing.config import ctx_list
from nnvm.testing.check_computation import check_function

def test_relu():
    x = sym.Variable("x")
    y = sym.relu(sym.leaky_relu(x, alpha=0.3) - 0.2)

    def forward(x):
        x = (x < 0) * x * 0.3 + (x > 0) * x - 0.2
        return (x > 0) * x

    def backward(head_grads, x):
        sub = (x < 0) * x * 0.3 + (x > 0) * x - 0.2
        return [(sub > 0).astype("float") * \
                ((x > 0).astype("float") + 0.3 * (x < 0).astype("float")) * head_grads]

    dtype = "float32"
    dshape = (1, 3, 32, 32)
    inputs = [(x, dshape)]
    check_function(y, inputs, forward, backward, dtype=dtype)

def test_prelu_nchw():
    x = sym.Variable("x")
    a = sym.Variable("a")
    y = sym.prelu(data=x, alpha=a)

    def forward(x, a):
        return (x < 0) * (x * a.reshape(3, 1, 1)) + (x>=0) * x

    dtype = "float32"
    dshape_x = (1, 3, 32, 32)
    dshape_w = (3,)

    inputs = [
        (x, dshape_x),
        (a, dshape_w)
    ]
    check_function(y, inputs, forward, dtype=dtype)

def test_prelu_nhwc():
    x = sym.Variable("x")
    a = sym.Variable("a")
    y = sym.prelu(data=x, alpha=a, axis=3)

    def forward(x, a):
        return (x < 0) * (x * a.reshape(1, 1, 3)) + (x>=0) * x

    dtype = "float32"
    dshape_x = (1, 32, 32, 3)
    dshape_w = (3,)

    inputs = [
        (x, dshape_x),
        (a, dshape_w)
    ]


    check_function(y, inputs, forward, dtype=dtype)

def test_sym_scalar_pow():
    scalar = 3
    x = sym.Variable("x")
    y = x**scalar

    def forward(x):
        return x**scalar

    def backward(head_grads, x):
        return [scalar * x**(scalar -  1) * head_grads]

    dtype = "float32"
    dshape = (1, 3, 32, 32)
    inputs = [(x, dshape)]
    check_function(y, inputs, forward, backward, dtype=dtype)


def test_scalar_sym_pow():
    scalar = 3
    x = sym.Variable("x")
    y = scalar**x

    def forward(x):
        return scalar**x

    def backward(head_grads, x):
        return [np.log(scalar) * scalar**x * head_grads]

    dtype = "float32"
    dshape = (1, 3, 32, 32)
    inputs = [(x, dshape)]
    check_function(y, inputs, forward, backward, dtype=dtype)


def test_exp():
    x = sym.Variable("x")
    y = sym.exp(x)

    def forward(x):
        return np.exp(x)

    def backward(head_grads, x):
        return [np.exp(x) * head_grads]

    dtype = "float32"
    dshape = (1, 3, 32, 32)
    inputs = [(x, dshape)]
    check_function(y, inputs, forward, backward, dtype=dtype)


def test_log():
    x = sym.Variable("x")
    y = sym.log(x)

    def forward(x):
        return np.log(x)

    def backward(head_grads, x):
        return [1. / x * head_grads]

    dtype = "float32"
    dshape = (1, 3, 32, 32)
    inputs = [(x, dshape)]
    check_function(y, inputs, forward, backward, in_range=(0.001, 2.0), dtype=dtype)


def test_tanh():
    x = sym.Variable("x")
    y = sym.tanh(x)

    def forward(x):
        return np.sinh(x) / np.cosh(x)

    def backward(head_grads, x):
        y_np = forward(x)
        return [(1 - y_np**2) * head_grads]

    dtype = "float32"
    dshape = (1, 3, 32, 32)
    inputs = [(x, dshape)]
    check_function(y, inputs, forward, backward, dtype=dtype)


def test_sigmoid():
    x = sym.Variable("x")
    y = sym.sigmoid(x)

    def forward(x):
        return 1.0 / (1.0 + np.exp(-x))

    def backward(head_grads, x):
        y_np = forward(x)
        return [y_np *(1 - y_np) * head_grads]

    dtype = "float32"
    dshape = (1, 3, 32, 32)
    inputs = [(x, dshape)]
    check_function(y, inputs, forward, backward, dtype=dtype)


def test_softmax():
    x = sym.Variable("x")
    y = sym.softmax(x)

    def forward(x):
        return topi.testing.softmax_python(x)

    def backward(head_grads, x):
        y = topi.testing.softmax_python(x)
        grad = y * (head_grads - np.sum(y * head_grads, axis=1, keepdims=True))
        return [grad]

    dtype = "float32"
    dshape = (10, 1000)
    inputs = [(x, dshape)]
    check_function(y, inputs, forward, backward, dtype=dtype)


def test_log_softmax():
    x = sym.Variable("x")
    y = sym.log_softmax(x)

    def forward(x):
        return topi.testing.log_softmax_python(x)

    def backward(head_grads, x):
        y = topi.testing.log_softmax_python(x)
        grad = head_grads - np.exp(y) * np.sum(head_grads, axis=1, keepdims=True)
        return [grad]

    dtype = "float32"
    dshape = (10, 1000)
    inputs = [(x, dshape)]
    check_function(y, inputs, forward, backward, dtype=dtype)


def test_dense():
    x = sym.Variable("x", shape=(10, 100))
    w = sym.Variable("dense_weight", shape=(3, 100))
    b = sym.Variable("dense_bias", shape=(3,))
    y = sym.dense(x, w, b, use_bias=True, units=3, name="dense")
    y = sym.flatten(y)

    def forward(x, dense_weight, dense_bias):
        return np.dot(x, dense_weight.T) + dense_bias
    dtype = "float32"
    inputs = [
        (x, (10, 100)),
        (w, (3, 100)),
        (b, (3,))
    ]
    check_function(y, inputs, forward, dtype=dtype)


def test_batchnorm():
    x = sym.Variable("x")
    beta = sym.Variable("beta")
    gamma = sym.Variable("gamma")
    moving_var = sym.Variable("moving_var")
    moving_mean = sym.Variable("moving_mean")
    eps = 1e-5
    y = sym.batch_norm(
        x, gamma, beta, moving_mean, moving_var, epsilon=eps)

    def forward(x, gamma, beta, moving_mean, moving_var):
        return (x - moving_mean) / np.sqrt(moving_var + eps) * gamma + beta

    dtype = "float32"
    inputs = [
        (x, (10, 20)),
        (gamma, (20,)),
        (beta, (20,)),
        (moving_mean, (20,)),
        (moving_var, (20,))
    ]

    check_function(y, inputs, forward, in_range=(0.001, 1.0), dtype=dtype)


def verify_concatenate(ishape, axis):
    x = [sym.Variable("x%d" % i) for i in range(len(ishape))]
    y = sym.concatenate(*x, axis=axis) + 1
    dtype = "float32"
    for target, ctx in ctx_list():
        # set input
        data = []
        for i, shape in enumerate(ishape):
            data.append(np.random.uniform(size=shape).astype(dtype))
        pdict = {"x%d" % i :  v for i, v in enumerate(data)}
        shape = {"x%d" % i :  v.shape for i, v in enumerate(data)}
        graph, lib, _ = nnvm.compiler.build(y, target, shape)
        m = graph_runtime.create(graph, lib, ctx)
        m.run(**pdict)
        out_np = np.concatenate(data, axis=axis) + 1
        out = m.get_output(0, tvm.nd.empty(out_np.shape))
        np.testing.assert_allclose(out.asnumpy(), out_np, atol=1e-5, rtol=1e-5)


def test_concatenate():
    verify_concatenate([(2, 3, 4), (1, 3, 4)], axis=0)
    verify_concatenate([(2, 4), (2, 7)], axis=1)


def verify_split(ishape, indices_or_sections, axis):
    x = sym.Variable("x")
    y = sym.split(x, indices_or_sections=indices_or_sections, axis=axis)
    dtype = "float32"
    x_np = np.random.uniform(size=ishape).astype(dtype)
    res = np.split(x_np, indices_or_sections, axis=axis)
    for target, ctx in ctx_list():
        # set input
        graph, lib, _ = nnvm.compiler.build(y, target, {"x": ishape})
        m = graph_runtime.create(graph, lib, ctx)
        m.run(x=x_np)
        for i, arr  in enumerate(res):
            out = m.get_output(i, tvm.nd.empty(arr.shape))
            np.testing.assert_allclose(out.asnumpy(), arr, atol=1e-5, rtol=1e-5)


def test_split():
    verify_split((2, 3), 2, axis=0)
    verify_split((5, 3), [3], axis=0)
    verify_split((5, 9, 3), [3, 4], axis=1)

def verify_strided_slice(ishape, begin, end, strideinp=None):
    stride = strideinp if strideinp else [1, 1, 1]
    x = sym.Variable("x")
    if strideinp:
        y = sym.strided_slice(x, begin = begin, end = end, stride = stride) + 1
    else:
        y = sym.strided_slice(x, begin = begin, end = end) + 1
    x_np = np.random.uniform(size=ishape).astype("float32")
    for i in range(len(begin), 3):
        begin.append(0)
    for i in range(len(end), 3):
        end.append(ishape[i])
    def test_forward(x, begin, end, stride):
        return x[begin[0]:end[0]:stride[0],
                    begin[1]:end[1]:stride[1], begin[2]:end[2]:stride[2]] + 1

    for target, ctx in ctx_list():
        # set input
        graph, lib, _ = nnvm.compiler.build(y, target, {"x": ishape})
        m = graph_runtime.create(graph, lib, ctx)
        m.run(x=x_np)
        res = test_forward(x_np, begin, end, stride)
        out = m.get_output(0, tvm.nd.empty(res.shape))
        np.testing.assert_allclose(out.asnumpy(), res, atol=1e-5, rtol=1e-5)

def test_strided_slice():
    verify_strided_slice((3, 4, 3), [0, 0, 0], [4, -5, 4], [1, -1, 2])
    verify_strided_slice((3, 4, 3), [1, 1, 0], [4, 4, 3], [2, 1, 1])
    verify_strided_slice((3, 4, 3), [1, -1, 0], [4, -5, 3], [2, -1, 1])
    verify_strided_slice((3, 4, 3), [1, 0, 0], [2, 2, 3], [1, 1, 2])
    verify_strided_slice((3, 4, 3), [1, -1, 0], [2, -3, 3], [1, -1, 1])
    verify_strided_slice((3, 4, 3), [1, 1, 0], [4, 4, 3])
    verify_strided_slice((3, 4, 3), [1, 1, 0], [4, 1000, 3])
    verify_strided_slice((3, 4, 3), [1, 1, 0], [4, 4])
    verify_strided_slice((3, 4, 3), [1, 1], [4, 4, 3])

def verify_take(src_shape, indices_src, axis=None):
    src_dtype = "float32"
    indices_dtype = "int32"
    indices_src = np.array(indices_src, dtype=indices_dtype)
    a = sym.Variable("a")
    indices = sym.Variable("indices")
    y = sym.take(a, indices, axis=axis)
    for target, ctx in ctx_list():
        # set input
        shape_dict = {"a":src_shape, "indices":indices_src.shape}
        type_dict = {"a":src_dtype, "indices":indices_dtype}
        graph, lib, _ = nnvm.compiler.build(y, target, shape=shape_dict, dtype=type_dict)
        m = graph_runtime.create(graph, lib, ctx)

        shape_size = 1
        for i in range(len(src_shape)):
            shape_size = shape_size * src_shape[i]
        a_src = np.arange(shape_size, dtype=src_dtype).reshape((src_shape))
        out_np = np.take(a_src, indices_src, axis=axis)
        m.run(a=a_src, indices=indices_src)
        out = m.get_output(0, tvm.nd.empty(out_np.shape, dtype=src_dtype))
        np.testing.assert_allclose(out.asnumpy(), out_np, atol=1e-5, rtol=1e-5)

def test_take():
    verify_take((4,), [1])
    verify_take((4,), [[0,1,2,3]])
    verify_take((3,3,3), [[11,25]])
    verify_take((4,), [[0,1],[2,3]])
    verify_take((4,), [1], 0)
    verify_take((2,2), [[[1,0],[0,1]]], 0)
    verify_take((2,2), [[[1,0],[0,1]]], 1)
    verify_take((4,3,5,6), [[2,1,0,0]], -2)


def verify_squeeze(dshape, axis):
    x = sym.Variable("x")
    if axis is not None:
        y = sym.squeeze(x, axis=axis)
    else:
        y = sym.squeeze(x)
    y = y + 1

    def forward(x):
        return np.squeeze(x, axis=axis) + 1

    def backward(head_grads, x):
        return [np.reshape(head_grads, x.shape)]

    dtype = "float32"
    inputs = [(x, dshape)]
    check_function(y, inputs, forward, backward, dtype=dtype)


def test_squeeze():
    verify_squeeze((1, 3, 2, 5), None)
    verify_squeeze((1, 3, 1), axis=0)
    verify_squeeze((1, 3, 2, 5, 1), axis=-1)


def test_pad():
    x = sym.Variable("x")
    y = sym.pad(x, pad_width=((0, 0), (0, 0), (0, 1), (2, 3)), pad_value=1.)

    def forward(x):
        return np.pad(x,
                      pad_width=((0, 0), (0, 0), (0, 1), (2, 3)),
                      mode='constant', constant_values=1.)

    dtype = "float32"
    inputs = [(x, (1, 3, 28, 28))]
    check_function(y, inputs, forward, dtype=dtype)

def verify_lrn(ishape, size, axis, bias, alpha, beta):
    x = sym.Variable("x")
    y = sym.lrn(x, size=size, axis=axis, bias=bias, alpha=alpha, beta=beta)
    dtype = "float32"
    x_np = np.random.uniform(size=ishape).astype(dtype)

    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, {"x": ishape})
        m = graph_runtime.create(graph, lib, ctx)
        m.run(x=x_np)
        out = m.get_output(0, tvm.nd.empty(ishape))
        out_np = topi.testing.lrn_python(x_np, size, axis, bias, alpha, beta)
        np.testing.assert_allclose(out.asnumpy(), out_np, atol=1e-5, rtol=1e-5)

    #Checking LRN op followed by elementwise op relu
    z = sym.relu(y)
    x_np = np.random.uniform(low=-10.0, high=10.0, size=ishape).astype(dtype)
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(z, target, {"x": ishape})
        m = graph_runtime.create(graph, lib, ctx)
        m.run(x=x_np)
        out = m.get_output(0, tvm.nd.empty(ishape))
        out_np = topi.testing.lrn_python(x_np, size, axis, bias, alpha, beta)
        out_np = (out_np > 0) * out_np
        np.testing.assert_allclose(out.asnumpy(), out_np, atol=1e-5, rtol=1e-5)

def verify_l2_normalize(ishape, eps, axis):
    x = sym.Variable("x")
    y = sym.l2_normalize(x, eps=eps, axis=axis)
    dtype = "float32"
    x_np = np.random.uniform(size=ishape).astype(dtype)

    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(y, target, {"x": ishape})
        m = graph_runtime.create(graph, lib, ctx)
        m.run(x=x_np)
        out = m.get_output(0, tvm.nd.empty(ishape))
        out_np = topi.testing.l2_normalize_python(x_np, eps, axis)
        np.testing.assert_allclose(out.asnumpy(), out_np, atol=1e-5, rtol=1e-5)

    #Checking L2 normalization op followed by elementwise op relu
    z = sym.relu(y)
    x_np = np.random.uniform(low=-10.0, high=10.0, size=ishape).astype(dtype)
    for target, ctx in ctx_list():
        graph, lib, _ = nnvm.compiler.build(z, target, {"x": ishape})
        m = graph_runtime.create(graph, lib, ctx)
        m.run(x=x_np)
        out = m.get_output(0, tvm.nd.empty(ishape))
        out_np = topi.testing.l2_normalize_python(x_np, eps, axis)
        out_np = (out_np > 0) * out_np
        np.testing.assert_allclose(out.asnumpy(), out_np, atol=1e-5, rtol=1e-5)

def test_lrn():
    verify_lrn((1, 3, 20, 20), 3, 1, 1.0, 1.0, 0.5)
    verify_lrn((1, 3, 20, 20), 3, 1, 2.0, 1.0, 0.75)

def test_l2_normalize():
    verify_l2_normalize((1, 3, 20, 20), 0.001, (1,))
    verify_l2_normalize((1, 3, 20, 20), 0.001, (1, 2))

if __name__ == "__main__":
    test_split()
    test_concatenate()
    test_log_softmax()
    test_batchnorm()
    test_dense()
    test_relu()
    test_prelu_nchw()
    test_prelu_nhwc()
    test_sym_scalar_pow()
    test_scalar_sym_pow()
    test_exp()
    test_log()
    test_tanh()
    test_sigmoid()
    test_softmax()
    test_squeeze()
    test_pad()
    test_take()
    test_lrn()
    test_l2_normalize()
    test_strided_slice()
