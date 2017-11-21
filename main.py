import numpy as np
import cupy as cp
from PIL import Image
import math
import fnmatch
import os
from os.path import join, relpath
from glob import glob

import argparse
from sklearn.model_selection import train_test_split

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset

#問題データセット
queData = []
#解凍データスェット
ansData = []
#学習用データのディレクトリ
tdPath = 'traindata'
#キャラ(切り出す画像データの単位)サイズ
charSize = 16*2
#色相分の水増し数(1,2,3,6,12)
hs = 6

xp = np

#学習用クラス
class ADPPD(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(ADPPD, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))#入力をl1で変換、さらに活性化関数で変換
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

#画像を水増しし、切り出し、データセットへ追加する
def appendModel(inImg,inData):
    img = [[0 for i in range(8)] for j in range(6)]
    img[0][0] = inImg
    img[0][4] = img[0][0].transpose(Image.ROTATE_90)

    for i in range(0, 8, 4):
        img[0][i+1] = img[0][i].transpose(Image.FLIP_LEFT_RIGHT)
        img[0][i+2] = img[0][i].transpose(Image.FLIP_TOP_BOTTOM)
        img[0][i+3] = img[0][i+1].transpose(Image.FLIP_TOP_BOTTOM)
    for j in range(0,hs):
        for i in range(0, 8):#反回転切り替え
            h, s, v = img[0][i].split()
            j2 = (255/hs)*j
            h2 = h.point(lambda h: round((h+j2)%255,0))
            img[j][i] = Image.merge("HSV", (h2 , s, v))
            for yCr in range(0, img[j][i].size[1], int(charSize/2)):
                for xCr in range(0, img[j][i].size[0], int(charSize/2)):
                    imgStr = img[j][i].crop((xCr, yCr, (xCr+charSize), (yCr+charSize)))
                    """
                    h,s,v = imgStr.split()
                    hAdd = xp.asarray(xp.float32(h)/255.0)
                    sAdd = xp.asarray(xp.float32(s)/255.0)
                    vAdd = xp.asarray(xp.float32(v)/255.0)
                    addData = xp.asarray([hAdd, sAdd, vAdd])
                    """
                    addData = xp.asarray(imgStr).transpose(2,0,1).astype(xp.float32)/255.
                    inData.append(addData)

#画像領域をキャラの倍数に拡大
def fixImg(inFile,outFile):
    queImg = Image.open(inFile).convert("HSV")
    ansImg = Image.open(outFile).convert("HSV")

    xChar = math.ceil(queImg.size[0]/charSize)
    yChar = math.ceil(queImg.size[1]/charSize)

    xFix = xChar*charSize
    yFix = yChar*charSize

    size = (xFix,yFix)

    xSt = math.floor((xFix-queImg.size[0])/2)
    ySt = math.floor((yFix-queImg.size[1])/2)

    start = (xSt,ySt)

    white =(0,0,255)

    queImg2 = Image.new('HSV',size,white)
    queImg2.paste(queImg, start)
    appendModel(queImg2, queData)
    ansImg2 = Image.new('HSV',size,white)
    ansImg2.paste(ansImg, start)
    appendModel(ansImg2, ansData)

def main():
    allList = [relpath(x, tdPath) for x in glob(join(tdPath, '*'))]
    #画像ファイルからqueデータを検索
    inList = fnmatch.filter(allList,"*_i.bmp")
    #queデータと対になるansデータを検索
    for i in range(0, len(inList)):
        t1 = tdPath+'/'+inList[i][:-6]+'_i.bmp'
        t2 = tdPath+'/'+inList[i][:-6]+'_o.bmp'
        if(os.path.exists(t2) == True):
            fixImg(t1,t2)
    #ここから、chainer絡みの記述
    #初期設定
    parser = argparse.ArgumentParser(description='Chainer')
    parser.add_argument('--batchsize', '-b', type=int, default=20,
                        help='バッチサイズ')
    parser.add_argument('--epoch', '-e', type=int, default=50,
                        help='中間層の数')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPUの有無')
    parser.add_argument('--out', '-o', default='result',
                        help='リサルトファイルのフォルダ')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=4092,
                        help='中間層の数')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    ##MLPをここで引っ張る
    model = L.Classifier(ADPPD(args.unit, 3072))#out-10種類(0-9の数字判別のため)
    model.compute_accuracy = False
    #GPU有無の判別
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
        xp = cp

    ##optimizerのセット
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    ##データセットをchainerにセット
    que_train, que_test, ans_train, ans_test = train_test_split(queData, ansData, test_size=0.2)
    train = tuple_dataset.TupleDataset(que_train, ans_train)
    test = tuple_dataset.TupleDataset(que_test, ans_test)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    ##updater=重みの調整、今回はStandardUpdaterを使用
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    ##updaterをtrainerにセット
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    ##評価の際、Evaluatorを使用
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))


    ##途中経過の表示用の記述
    trainer.extend(extensions.dump_graph('main/loss'))
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
    trainer.extend(extensions.LogReport())
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    #中断データの有無、あれば続きから
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
    ##実験開始、trainerにお任せ
    trainer.run()


if __name__ == '__main__':
    main()