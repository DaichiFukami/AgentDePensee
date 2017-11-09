import numpy as np
import cupy as cp
from PIL import Image
import math

#xp = np
xp = cp

import chainer
from chainer import cuda
from chainer import optimizers
import chainer.functions as F
from chainer.links import caffe
from sympy.printing.pretty.pretty_symbology import xstr

