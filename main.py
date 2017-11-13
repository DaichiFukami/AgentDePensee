import numpy as np
import cupy as cp
from PIL import Image
import math
import fnmatch
import os
from os.path import join, relpath
from glob import glob

import chainer
from chainer import cuda
from chainer import optimizers
import chainer.functions as F
from chainer.links import caffe
from sympy.printing.pretty.pretty_symbology import xstr

#問題データ
queData = []
#解凍データ
ansData = []
#学習用データのディレクトリ
tdPath = 'traindata'
#キャラ(切り出す画像データの単位)サイズ
charSize = 16*2
xp = np

def appendModel(inImg,inData):
    img = [0,1,2,3,4,5,6,7]
    img[0] = inImg
    img[4] = img[0].transpose(Image.ROTATE_90)
    for i in range(0, 8, 4):
        img[i+1] = img[i].transpose(Image.FLIP_LEFT_RIGHT)
        img[i+2] = img[i].transpose(Image.FLIP_TOP_BOTTOM)
        img[i+3] = img[i+1].transpose(Image.FLIP_TOP_BOTTOM)
    for i in range(0, 8):#反回転切り替え
        for yCr in range(0, img[i].size[1], int(charSize/2)):
            for xCr in range(0, img[i].size[0], int(charSize/2)):
                imgStr = img[i].crop((xCr, yCr, (xCr+charSize), (yCr+charSize)))
                h,s,v = imgStr.split()
                hAdd = xp.asarray(xp.float32(h)/360.0)
                sAdd = xp.asarray(xp.float32(s)/255.0)
                vAdd = xp.asarray(xp.float32(v)/255.0)
                addData = xp.asarray([hAdd, sAdd, vAdd])
                inData.append(addData)

def fixImg(inFile,outFile):
    queImg = Image.open(inFile).convert("HSV")
    ansImg = Image.open(outFile).convert("HSV")
    x = queImg.size[0]
    y = queImg.size[1]

    xChar = math.ceil(x/charSize)
    yChar = math.ceil(y/charSize)

    xFix = xChar*charSize
    yFix = yChar*charSize

    size = (xFix,yFix)
    white =(0,0,255)
    xSt = math.floor((xFix-x)/2)
    ySt = math.floor((yFix-y)/2)

    start = (xSt,ySt)

    queImg2 = Image.new('HSV',size,white)
    queImg2.paste(queImg, start)
    appendModel(queImg2, queData)
    ansImg2 = Image.new('HSV',size,white)
    ansImg2.paste(ansImg, start)
    appendModel(ansImg2, ansData)


allList = [relpath(x, tdPath) for x in glob(join(tdPath, '*'))]
inList = fnmatch.filter(allList,"*_i.bmp")
for i in range(0, len(inList)):
    t1 = tdPath+'/'+inList[i][:-6]+'_i.bmp'
    t2 = tdPath+'/'+inList[i][:-6]+'_o.bmp'
    if(os.path.exists(t2) == True):
        fixImg(t1,t2)
print('ここまでできた')