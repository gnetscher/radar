from radarUtils import plotRadar
import ipdb
import argparse
import ffmpeg
import os

import caffe
from caffe.proto import caffe_pb2
from caffe import layers as L
from caffe import params as P

"""

Network Definitions

"""
weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]
frozen_param = [dict(lr_mult=0)] * 2

zero_filler     = dict(type='constant', value=0)
msra_filler     = dict(type='msra')
uniform_filler  = dict(type='uniform', min=-0.1, max=0.1)
fc_filler       = dict(type='gaussian', std=0.005)
conv_filler     = dict(type='msra')

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=learned_param, weight_filler=conv_filler, bias_filler=zero_filler):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=nout, pad=pad, 
			 group=group, param=param, weight_filler=weight_filler, bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout, param=learned_param, weight_filler=fc_filler, bias_filler=zero_filler):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,weight_filler=weight_filler, bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1, train=False):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


def defineSolver(args):
    solver = caffe_pb2.SolverParameter()
    #TODO: set solver parameters...

def defineNetwork(args, imageFiles, radarData):
    net = caffe.NetSpec()
    #TODO: create data layer or setup image/label txt files
    net.data, net.label = None, None
    net.conv1, net.relu1 = conv_relu(net.data, 11, 96, stride=4)
    net.pool1 = max_pool(net.relu1, 3, stride=2)

    net.conv2, net.relu2 = conv_relu(net.pool1, 5, 256, pad=2)
    net.pool2 = max_pool(net.relu2, 3, stride=2)


    #TODO: continue making net + add late fusion
    #TODO: read in 

"""

Training Code

"""



# TODO: just take timestamps for the labels

def saveFrames(frames, outputDir, videoFile):
    vidName = videoFile.split('/')[-1].split('.')[0]
    for i,v in enumerate(frames):
        savePath = outputDir + vidName + '_%s.jpg' % (i)
        v.save(savePath)
    return [outputDir + x for x in os.listdir(outputDir)]

def main(args):
    frames = ffmpeg.extract(args.video)
    if not os.path.exists(args.outputDir):
        os.mkdir(args.outputDir)
        os.mkdir(args.outputDir + 'data/')
    
    imageFiles = saveFrames(frames, args.outputDir + 'data/', args.video)
    radarData = None #TODO: read in radar data

    if args.gpu >= 0:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu)
    else:
        caffe.set_mode_cpu()

    train_net, test_net = defineNetwork(args, imageFiles, radarData)
    #TODO: create prototxt files
    #TODO: call training command for protoxt



def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='test29.mp4', help='RGB Video File')
    #TODO: change this to include multiple radar files for each of the images
    parser.add_argument('--radar', type=str, default='nokia_data/image3d_2017.01.12_10.29.mat', help='file with radar information')
    parser.add_argument('--outputDir', type=str,default='output/', help='dir to store random processing output')
    parser.add_argument('--gpu', type=int, default=0, help='GPU used to train network; set to -1 for CPU training')
    return parser.parse_args()

if __name__=="__main__":
    args = parseArgs()
    plotRadar('nokia_data/image3d_2017.01.12_10.29.mat', 'test29.mp4')
    main(args)
    
