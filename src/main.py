from radarUtils import plot_radar
import ipdb
import argparse
import ffmpeg
import os
import os.path as osp

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

def max_pool(bottom, ks, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


def defineSolver(args):
    solver = caffe_pb2.SolverParameter()
    solver.train_net = args.train_net
    if args.test_net:
    	solver.test_net = args.test_net
   	solver.test_iter = 50
    solver.test_interval = 100
    solver.iters_size = 1
    solver.type = "SGD"
    solver.base_lr = 0.001
    solver.lr_policy = 'step'
    solver.gamma = 0.1
    solver.stepsize = 20000
    solver.max_iter = 100000
    solver.momentum = 0.9
    solver.weight_decay = 0.0005
    solver.display = 50
    solver.average_loss = 1
    solver.snapshot = 5000
    solver.snapshot_prefix = args.snapshotDir
    if not os.path.exists(args.snapshotDir):
	os.path.mkdir(args.snapshot_dir)
    return solver

#TODO: set solver parameters...

def defineNetwork(args, imageFiles, radarData, training=False):
    net = caffe.NetSpec()
    #TODO: create data layer or setup image/label txt files
    net.data, net.label = None, None
    net.conv1, net.relu1 = conv_relu(net.data, 3, 64, pad=1)
    net.pool1 = max_pool(net.relu1, 2)

    net.conv2, net.relu2 = conv_relu(net.data, 3, 128, pad=1)
    net.pool2 = max_pool(net.relu2, 2)
    
    net.conv3, net.relu3 = conv_relu(net.data, 3, 256, pad=1)
    net.pool3 = max_pool(net.relu3, 2)

    net.fc4, net.relu4 = fc_relu(net.pool3, 1024)
    net.drop4 = L.Dropout(net.relu4, dropout_ratio=0.5, in_place=True)

    #TODO: incorporate radar features...
    net.fc5, net.relu5 = fc_relu(net.pool4, 512)
    net.drop5 = L.Dropout(net.relu5, dropout_ratio=0.5, in_place=True)

    net.final = L.InnerProduct(net.drop5, num_output=3, param=learned_param)
    net.loss = L.SoftmaxWithLoss(net.final, net.label)
    if training:
    	net.acc = L.Accuracy(net.final, net.label)
    return net.to_proto()

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
   
    args.train_net = args.outputDir + 'train_net.prototxt'
    args.test_net  = args.outputDir + 'test_net.prototxt'
    args.solver_file = args.outputDir + 'solver.prototxt' 

    imageFiles = saveFrames(frames, args.outputDir + 'data/', args.video)
    radarData = None #TODO: read in radar data

    if args.gpu >= 0:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu)
    else:
        caffe.set_mode_cpu()

    solver = defineSolver(args)
    trainNet = defineNetwork(args, imageFiles, radarData, training=True)
    testNet = defineNetwork(args, imageFiles, radarData, training=False)
    
    with open(args.train_net,'w') as f:
	f.write(str(trainNet))
    with open(args.test_net,'w') as ft:
	ft.write(str(testNet))
    with open(args.solver_file,'w') as fs:
	fs.write(str(solver))


    #TODO: call training command for protoxt



def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='test29.mp4', help='RGB Video File')
    #TODO: change this to include multiple radar files for each of the images
    parser.add_argument('--radar', type=str, default='nokia_data/image3d_2017.01.12_10.29.mat', help='file with radar information')
    parser.add_argument('--outputDir', type=str,default='output/', help='dir to store random processing output')
    parser.add_argument('--snapshotDir',type=str, default='snapshsots/', help='where to store training snapshots')
    parser.add_argument('--gpu', type=int, default=0, help='GPU used to train network; set to -1 for CPU training')
    return parser.parse_args()

if __name__=="__main__":
    args = parseArgs()
    baseDir = '/mnt/HardDrive/common/nokia_radar/sleeplab'
    outDir  = '../out'
    plot_radar(osp.join(baseDir, 'image3d_2017.01.12_10.29.mat'), osp.join(outDir, 'test29.mp4'))
    # main(args)
    
