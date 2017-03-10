from radarUtils import plot_radar
import ipdb
from ns_backend import *
import numpy as np
from chainer.objects import data_objects as dj
import argparse
import os
import os.path as osp
import subprocess

import caffe
from caffe.proto import caffe_pb2
from caffe import layers as L
from caffe import params as P
import data_layer

print caffe.__file__
print dir(data_layer)

"""
Network Definitions
"""
LABELMAP = {"Sitting": 0, "Standing": 1, "Fall": 2, "Walking": 1, "On the ground": 2, "Lying down": 4}
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
  bn = L.BatchNorm(conv, param=[dict(lr_mult=0)]*3)
  return conv, L.ReLU(bn, in_place=True)

def fc_relu(bottom, nout, param=learned_param, weight_filler=fc_filler, bias_filler=zero_filler):
  fc = L.InnerProduct(bottom, num_output=nout, param=param,weight_filler=weight_filler, bias_filler=bias_filler)
  bn = L.BatchNorm(fc, param=[dict(lr_mult=0)]*3)
  return fc, L.ReLU(bn, in_place=True)

def max_pool(bottom, ks, stride=2):
  return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


def define_solver(args):
  solver = caffe_pb2.SolverParameter()
  solver.train_net = args.trainNet
  solver.test_net.append(args.testNet)
  solver.test_iter.append(50)
  solver.test_interval = 100
  solver.iter_size = 1
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
  solver.snapshot_prefix = osp.join(os.getcwd(),args.snapshotDir)
  if not os.path.exists(args.snapshotDir):
    os.mkdir(args.snapshotDir)
  return solver


def define_network(args, imageFile, videos, radarFiles, training=False):
  net = caffe.NetSpec()
  
  #setting up data layer..
  mean = [117.193,  117.673,  114.125] 
  transformParam = dict(mirror=training, mean_value = mean)
  pydataParams = dict(radar_files = radarFiles, videos = videos, batch_size = args.batchSize)
  # TODO: add shuffling capability
  net.data, net.label = L.ImageData(transform_param = transformParam, source=imageFile, shuffle=False, batch_size=args.batchSize, ntop=2)
  net.radar = L.Python(module='data_layer', layer='RadarDataLayer', param_str=str(pydataParams), ntop=1)
  
  net.conv1_1, net.relu1_1 = conv_relu(net.data, 3, 32)
  net.conv1_2, net.relu1_2 = conv_relu(net.relu1_1, 3, 32)
  net.pool1 = max_pool(net.relu1_2, 2)

  net.conv2_1, net.relu2_1 = conv_relu(net.pool1, 3, 64)
  net.conv2_2, net.relu2_2 = conv_relu(net.relu2_1, 3, 64)
  net.pool2 = max_pool(net.relu2_2, 2)
    
  net.conv3_1, net.relu3_1 = conv_relu(net.pool2, 3, 128)
  net.conv3_2, net.relu3_2 = conv_relu(net.relu3_1, 3, 128)
  net.conv3_3, net.relu3_3 = conv_relu(net.relu3_2, 3, 128)
  net.pool3 = max_pool(net.relu3_3, 2)

  net.conv4_1, net.relu4_1 = conv_relu(net.pool3, 3, 256)
  net.conv4_2, net.relu4_2 = conv_relu(net.relu4_1, 3, 256)
  net.conv4_3, net.relu4_3 = conv_relu(net.relu4_2, 3, 256)
  net.pool4 = max_pool(net.relu4_3, 2)

  net.fc5, net.relu5 = fc_relu(net.pool4, 4096)
  net.concat = L.Concat(net.fc5, net.radar)
  net.drop5 = L.Dropout(net.concat, dropout_ratio=0.5, in_place=True)

  net.fc6, net.relu6 = fc_relu(net.drop5, 1024)
  net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)

  net.final = L.InnerProduct(net.drop5, num_output=4, param=learned_param)
  net.loss = L.SoftmaxWithLoss(net.final, net.label)

  if not training:
    net.acc = L.Accuracy(net.final, net.label)

  return net.to_proto()

"""
Training Code
"""
def create_data_txt(videos, stage='train'):
  fileName = "%s.txt" % (stage)
  frameFiles = []; frameLabels = []
    
  for v in videos:  
    if not v.is_saved_frames():
      v.save_frames()

    loc, glob = v.annotation
    if 'person' in loc:
      searchKey = 'person'
    elif 'Nokia' in loc:
      searchKey = 'Nokia'
    elif 'purple' in loc:
      searchKey = 'purple'
    elif 'Purple' in loc:
      searchKey = 'Purple'

    for k in loc[searchKey]:
      labels = loc[searchKey][k]['labels']
      for l in labels:
        if l in LABELMAP:
          frameFiles.append(v.get_frame_path(frameNum=k))
          frameLabels.append(LABELMAP[l])

  with open(fileName,'w') as f:
    for t in range(len(frameFiles)):
      f.write('%s %s\n' %(frameFiles[t], frameLabels[t]))
  return fileName  

def train(args):
  runstring = '/mnt/HardDrive/common/pkg/caffe/build/tools/caffe train -solver {} -gpu {} 2>&1 | tee -a {}'.format(args.solverFile, str(args.gpu), args.logFile)
  if args.weights:
    runstring = '/mnt/HardDrive/common/pkg/caffe/build/tools/caffe train -solver {} -gpu {} -weights {} 2>&1 | tee -a {}'.format(args.solverFile, str(args.gpu), args.weights, args.logFile)
  os.system(runstring)
  
def main(args):
  videos = NSVideo.objects.filter(sensor_id__startswith='Nokia')
  videos = [dj.Video(v) for v in videos if v.annotation_id]
  filteredVideos = []
  for v in videos:
    local, glob = v.annotation
    if local and ('101656' in v.local_path):
      filteredVideos.append(v)
  
  radarFiles = [osp.join(args.radarDir,'image3d_2017.01.12_10.%s.mat' % (str(int(v.local_path.split('.')[-2]) + 17))) for v in filteredVideos]
  
  for r,v in zip(radarFiles, filteredVideos):   
    if not osp.exists(r):
      radarFiles.remove(r)
      filteredVideos.remove(v)

  trainInd = np.random.choice(len(filteredVideos), int(0.8*len(filteredVideos)), replace=False)
  valInd = np.delete(np.arange(len(filteredVideos)), trainInd)

  trainVideos = [filteredVideos[x] for x in trainInd]
  valVideos = [filteredVideos[x] for x in valInd]

  trainTxt = create_data_txt(trainVideos, stage="train")
  valTxt = create_data_txt(valVideos, stage="val")

  if not os.path.exists(args.outputDir):
    os.mkdir(args.outputDir)
   
  args.trainNet = osp.join(os.getcwd(), args.outputDir + 'train_net.prototxt')
  args.testNet  = osp.join(os.getcwd(), args.outputDir + 'test_net.prototxt')
  args.solverFile = osp.join(os.getcwd(), args.outputDir + 'solver.prototxt')

  if args.gpu >= 0:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)
  else:
    caffe.set_mode_cpu()

  solver = define_solver(args)
  trainNet = define_network(args, trainTxt, trainVideos, radarFiles, training=True)
  testNet = define_network(args, valTxt, valVideos, radarFiles, training=False)
    
  with open(args.trainNet,'w') as f:
    f.write(str(trainNet))
  with open(args.testNet,'w') as ft:
    ft.write(str(testNet))
  with open(args.solverFile,'w') as fs:
    fs.write(str(solver))

  train(args)


def parseArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument('--radarDir', type=str, default='/mnt/HardDrive/common/nokia_radar/sleeplab', help='dir that contains radar files')
  parser.add_argument('--logFile', type=str, default='train.log')
  parser.add_argument('--outputDir', type=str,default='output/', help='dir to store random processing output')
  parser.add_argument('--weights', type=str, default=None, help="pretrained weights for loading")
  parser.add_argument('--snapshotDir',type=str, default='snapshots/', help='where to store training snapshots')
  parser.add_argument('--batchSize', type=int , default=64, help='batch size for training')
  parser.add_argument('--gpu', type=int, default=0, help='GPU used to train network; set to -1 for CPU training')
  return parser.parse_args()

if __name__=="__main__":
  args = parseArgs()
  main(args)

