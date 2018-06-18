import mxnet as mx
import numpy as np
import tvm
import nnvm

from os import mkdir
from os.path import abspath, join, dirname, isdir, isfile
from PIL import Image
from mxnet.gluon.model_zoo.vision import get_model
from mxnet.gluon.utils import download
from nnvm.compiler import build, save_param_dict
from pdb import set_trace

EXAMPLE_ROOT = abspath(join(dirname(__file__)))
BIN_DIR = join(EXAMPLE_ROOT, 'bin')
LIB_DIR = join(EXAMPLE_ROOT, 'lib')
DATA_DIR = join(EXAMPLE_ROOT, 'data')

def _transform_image(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image


def _download_image(img_dir):
    img_path = join(img_dir, 'cat.png')
    bin_img_path = join(img_dir, 'cat.bin')

    if not isdir(img_dir):
        mkdir(img_dir)
    if not isfile(img_path):
      download('https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true', img_path)

    img = Image.open(img_path).resize((224, 224))
    img = _transform_image(img)
    img.astype('float32').tofile(bin_img_path)
    shape_dict = {'data': img.shape}

    return shape_dict


def main():
    mx_model = get_model('resnet18_v1', pretrained=True)

    # load the model, input image, and imagenet classes
    shape_dict = _download_image(DATA_DIR)

    # convert the model, add a softmax
    sym, params = nnvm.frontend.from_mxnet(mx_model)
    sym = nnvm.sym.softmax(sym)

    # build the graph
    graph, lib, params = build(
        sym, "llvm --system-lib", shape_dict, params=params)

    # save the built graph
    if not isdir(LIB_DIR):
        mkdir(LIB_DIR)
    lib.save(join(LIB_DIR, 'deploy_lib.o'))
    with open(join(LIB_DIR, 'deploy_graph.json'), 'w') as f_graph_json:
        f_graph_json.write(graph.json())
    with open(join(LIB_DIR, 'deploy_params.bin'), 'wb') as f_params:
        f_params.write(save_param_dict(params))

if __name__ == '__main__':
    main()

