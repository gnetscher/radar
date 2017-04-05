from radarUtils import plot_radar
import ipdb
from ns_backend import *
import numpy as np
from chainer.objects import data_objects as dj
import argparse
import os, sys
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
#fc_filler       = dict(type='gaussian', std=0.005)
fc_filler       = dict(type='xavier')
conv_filler     = dict(type='msra')

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1, param=learned_param, weight_filler=conv_filler, bias_filler=zero_filler):
  conv = L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=nout, pad=pad, group=group, param=param, weight_filler=weight_filler, bias_filler=bias_filler)
  return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout, param=learned_param, weight_filler=fc_filler, bias_filler=zero_filler):
  fc = L.InnerProduct(bottom, num_output=nout, param=param,weight_filler=weight_filler, bias_filler=bias_filler)
  return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=2):
  return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


def define_solver(args):
  solver = caffe_pb2.SolverParameter()
  solver.train_net = args.trainNet
  solver.test_net.append(args.testNet)
  solver.test_iter.append(50)
  solver.test_interval = 1000
  solver.iter_size = 1
  solver.type = "SGD"
  solver.base_lr = 0.000001
  solver.lr_policy = 'step'
  solver.gamma = 0.1
  solver.stepsize = 20000
  solver.max_iter = 50000
  solver.momentum = 0.9
  solver.weight_decay = 0.0005
  solver.display = 50
  solver.average_loss = 1
  solver.snapshot = 10000
  solver.snapshot_prefix = osp.join(os.getcwd(),args.snapshotDir)
  if not os.path.exists(args.snapshotDir):
    os.mkdir(args.snapshotDir)
  return solver


def define_network(args, imageFile, vidIds, radarFiles, training=False):
  net = caffe.NetSpec()
  
  # Setting up data layer
  mean = [117.193,  117.673,  114.125] 
  transformParam = dict(mirror=training, mean_value = mean)
  pydataParams = dict(radar_files = radarFiles, videos = vidIds, batch_size = args.batchSize)
  
  net.data, net.label = L.ImageData(transform_param = transformParam, source=imageFile, shuffle=False, batch_size=args.batchSize, ntop=2)
  net.radar = L.Python(module='radarDataLayer', layer='RadarDataLayer', param_str=str(pydataParams), ntop=1)
  
  net.conv1, net.relu1 = conv_relu(net.data, 11, 96, stride=4)
  net.pool1 = max_pool(net.relu1, 3, stride=2)
  net.norm1 = L.LRN(net.pool1, local_size=5, alpha=1e-4, beta=0.75)

  net.conv2, net.relu2 = conv_relu(net.norm1, 5, 256, pad=2, group=2)
  net.pool2 = max_pool(net.relu2, 3, stride=2)
  net.norm2 = L.LRN(net.pool2, local_size=5, alpha=1e-4, beta=0.75)

  net.conv3, net.relu3 = conv_relu(net.norm2, 3, 384, pad=1)
  net.conv4, net.relu4 = conv_relu(net.relu3, 3, 384, pad=1, group=2)
  net.conv5, net.relu5 = conv_relu(net.relu4, 3, 256, pad=1, group=2)
  net.pool5 = max_pool(net.relu5, 3, stride=2)

  net.fc6_new, net.relu6_new = fc_relu(net.pool5, 4096)
  net.drop6 = L.Dropout(net.relu6_new, in_place=True)

  net.fc7_new = L.InnerProduct(net.drop6, num_output=4096, param=learned_param, weight_filler=fc_filler)
  net.concat = L.Concat(net.fc7_new, net.radar)
  net.relu7 = L.Concat(net.concat, in_place=True)
  net.drop7 = L.Dropout(net.relu7, in_place=True)
  
  net.final = L.InnerProduct(net.drop7, num_output=4, param=learned_param, weight_filler=fc_filler)
  net.loss = L.SoftmaxWithLoss(net.final, net.label)
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

    loc, glob = v.annotation.anns
    #elif 'person' in loc:
    #  searchKey = 'person'
    #elif 'Nokia' in loc:
    #  searchKey = 'Nokia'
    #elif 'purple' in loc:
    #  searchKey = 'purple'
    #elif 'Purple' in loc:
    #  searchKey = 'Purple'

    #for k in loc[searchKey]:
    #  labels = loc[searchKey][k]['labels']
    #  for l in labels:
    #    if l in LABELMAP:
    #      frameFiles.append(v.get_frame_path(frameNum=k))
    #      frameLabels.append(LABELMAP[l])

    for obj_key in loc:
      for frameInd in loc[obj_key]:
         labels = loc[obj_key][frameInd]['labels']
         if 'Person' in labels:
           for l in labels:
             if l in LABELMAP:
               frameFiles.append(v.get_frame_path(frameNum=frameInd))
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
  videos = NSVideo.objects.filter(sensor_id__startswith='Nokia').filter(annotation_id__isnull=False)
  videos = [dj.Video(v) for v in videos]
  filteredVideos = []
  for v in videos:
    local, glob = v.annotation.anns
    #if local and ('101656' in v.local_path):
    if local and ('imported' in v.local_path and 'nokoff' in v.local_path and '16-15' in v.local_path):
       filteredVideos.append(v)
  
  #radarFiles = [osp.join(args.radarDir,'image3d_2017.01.12_10.%s.mat' % (str(int(v.local_path.split('.')[-2]) + 17))) for v in filteredVideos]
  print [v.local_path for v in filteredVideos]
  fileLabs = [int(x.local_path.split('.')[-2]) for x in filteredVideos]
  ird = []
  for fl in fileLabs:
    if fl < 10:
      ird.append("0%s" %(str(fl)))
    else:
      ird.append(str(fl))
  print ird
  radarFiles = [osp.join(args.radarDir, 'image3d_2017.02.16_15.%s.mat' % (ir)) for ir in ird]


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

  trainIds = [str(v.video_id) for v in trainVideos]
  valIds = [str(v.video_id) for v in valVideos]

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
  trainNet = define_network(args, trainTxt, trainIds, radarFiles, training=True)
  testNet = define_network(args, valTxt, valIds, radarFiles, training=False)
  
  with open(args.trainNet,'w') as f:
    f.write(str(trainNet))
  with open(args.testNet,'w') as ft:
    ft.write(str(testNet))
  with open(args.solverFile,'w') as fs:
    fs.write(str(solver))

  train(args)


def parseArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument('--radarDir', type=str, default='/mnt/HardDrive/common/nokia_radar/office', help='dir that contains radar files')
  parser.add_argument('--logFile', type=str, default='train.log')
  parser.add_argument('--outputDir', type=str,default='output/', help='dir to store random processing output')
  parser.add_argument('--weights', type=str, default='/mnt/HardDrive/common/caffe_models/caffenet/bvlc_reference_caffenet.caffemodel', help="pretrained weights for loading")
  parser.add_argument('--snapshotDir',type=str, default='snapshots/', help='where to store training snapshots')
  parser.add_argument('--batchSize', type=int , default=32, help='batch size for training')
  parser.add_argument('--gpu', type=int, default=1, help='GPU used to train network; set to -1 for CPU training')
  return parser.parse_args()

if __name__=="__main__":
  args = parseArgs()
  main(args)

