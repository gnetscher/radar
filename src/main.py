from radarUtils import plot_radar
import ipdb
from ns_backend import *
from chainer.objects import data_objects as dj
import argparse
import os
import os.path as osp
import subprocess

import caffe
from caffe.proto import caffe_pb2
from caffe import layers as L
from caffe import params as P

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
  bn = L.BatchNorm(fc, param=[dict(lr_mult=0)]*3)
  return conv, L.ReLU(bn, in_place=True)

def fc_relu(bottom, nout, param=learned_param, weight_filler=fc_filler, bias_filler=zero_filler):
  fc = L.InnerProduct(bottom, num_output=nout, param=param,weight_filler=weight_filler, bias_filler=bias_filler)
  bn = L.BatchNorm(fc, param=[dict(lr_mult=0)]*3)
  return fc, L.ReLU(bn, in_place=True)

def max_pool(bottom, ks, stride=2):
  return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


def define_solver(args):
  solver = caffe_pb2.SolverParameter()
  solver.train_net = args.train_net
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


def define_network(args, imageFile, radarFile, training=False):
  net = caffe.NetSpec()
  
  #setting up data layer..
  mean = [104, 117, 123]
  batchSize = 64
  transformParam = dict(mirror=training, crop_size=args.crop, mean_value = mean)
  net.data, net.label = L.ImageData(transform_param = transformParam, source=imageFile, shuffle=training, batch_size=batch_size, ntop=2)
  net.radar = None
  
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
# creating train/val.txt for reading into caffe
def create_train_txt(videos, radar):
  trainName = "train.txt"; valName = "val.txt"
  frameFiles = []; frameLabels = []
     
  for i,v in enumerate(videos):
    if not v.is_saved_frames():
      v.save_frames()
    for i in range(v.frame_count):
      frameFiles.append(v.get_frame_path(frameNum=i))

    loc,glob = v.annotation
    for k in loc['person']:
      labels = loc['person'][k]['labels']
      for l in labels:
        if l in LABELMAP:
          frameLabels.append(LABELMAP[l])
    
    #TODO: get corresponding radar features
  trainSize = int(0.8*len(frameFiles))
  trainInd = np.random.choice(len(frameFiles), trainSize,replace=False)
  valInd = np.delete(np.arange(len(frameFiles)), trainInd)

  with open(trainName,'w') as f:
    for t in trainInd:
      f.write('%s %s\n' %(frameFiles[t], frameLabels[t]))
  with open(valName, 'w') as f:
    for t in valInd:
      f.write('%s %s\n' %(frameFiles[t], frameLabels[t]))
  return trainName, valName   

def train(args):
  cmd = ['caffe','train', '-solver', args.solverFile, '-gpu', args.gpu]
  if args.weights:
    cmd.extend(['-weights', args.weights])
  subprocess.call(cmd)
  
def main(args):
  videos = NSVideo.objects.get(sensor_id__startswith='Nokia')
  videos = [dj.Video(v) for v in videos if v.annotation_id]
  filteredVideos = []
  for v in videos:
    local, glob = v.annotation
    if local and ('104656' in v.local_path):
      filteredVideos.append(v)
  
  radarFiles = [args.radarDir + 'image3d_2017.01.12_10.%s.mat' %(str(int(av.local_path.split('.')[-2]) + 17)) for v in filteredVideos]
  for i in range(len(radarFiles)):
    if not osp.exists(radarFiles[i]):
      radarFiles.remove(radarFiles[i])
      filteredVideos.remove(filteredVideos[i])

  trainFile, trainRadar, valFile, valRadar = create_train_txt(filteredVideos, radarFiles)

  if not os.path.exists(args.outputDir):
    os.mkdir(args.outputDir)
   
  args.trainNet = args.outputDir + 'train_net.prototxt'
  args.testNet  = args.outputDir + 'test_net.prototxt'
  args.solverFile = args.outputDir + 'solver.prototxt' 

  if args.gpu >= 0:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)
  else:
    caffe.set_mode_cpu()

  solver = define_solver(args)
  trainNet = define_network(args, trainFile, trainRadar, training=True)
  testNet = define_network(args, valFile, valRadar, training=False)
    
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
  parser.add_argument('--outputDir', type=str,default='output/', help='dir to store random processing output')
  parser.add_argument('--weights', type=str, default=None, help="pretrained weights for loading")
  parser.add_argument('--snapshotDir',type=str, default='snapshsots/', help='where to store training snapshots')
  parser.add_argument('--gpu', type=int, default=0, help='GPU used to train network; set to -1 for CPU training')
  return parser.parse_args()

if __name__=="__main__":
  args = parseArgs()
  #outDir  = '../out'
  #plot_radar(osp.join(args.radarDir, 'image3d_2017.01.12_10.29.mat'), osp.join(outDir, 'test29.mp4'))
  main(args)
    
