import numpy as np
import cupy as cp
from PIL import Image
import math
import re
import fnmatch
import os
from os.path import join, relpath
from glob import glob
#xp = np
xp = cp

import chainer
from chainer import cuda
from chainer import optimizers
import chainer.functions as F
from chainer.links import caffe
from sympy.printing.pretty.pretty_symbology import xstr
"""
for inList in glob.iglob('traindata/*_i.bmp', recursive=True):
    print(inList)
"""

tdPath = 'traindata'
allList = [relpath(x, tdPath) for x in glob(join(tdPath, '*'))]
inList = fnmatch.filter(allList,"*_i.bmp")
for i in range(0, len(inList)):
    t2 = tdPath+'/'+inList[i][:-6]+'_o.bmp'
    if(os.path.exists(t2) == True):
