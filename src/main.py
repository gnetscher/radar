from radarUtils import plot_radar
import ipdb
from ns_backend import *
import numpy as np
import scipy.io as spio
import argparse
import os, sys
import os.path as osp
import subprocess
import caffe
from chainer.chainer_utils import io_utils
from caffe.proto import caffe_pb2
from caffe import layers as L
from caffe import params as P
from chainer.config import *

"""
Network Definitions
"""
LABELMAP = {"Sitting": 0, "Standing": 1, "Fall": 2, "Walking": 1, "On the ground": 2, "Lying down": 3}
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
  solver.max_iter = 20000
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
  transformParam = dict(mirror=training, mean_value = args.mean)
  pydataParams = dict(radar_files = radarFiles, videos = vidIds, batch_size = args.batchSize)
  
  net.data, net.label = L.ImageData(transform_param = transformParam, source=imageFile, shuffle=False, batch_size=args.batchSize, ntop=2)
  if args.expType != 'image':
    net.radar = L.Python(module='radarDataLayer', layer='RadarDataLayer', param_str=str(pydataParams), ntop=1)
 
  if args.expType == "joint" or args.expType == "image":
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
    
    if args.expType == "joint":
      net.concat = L.Concat(net.fc7_new, net.radar)
      net.relu7 = L.ReLU(net.concat, in_place=True)
    else:
      net.relu7 = L.ReLU(net.fc7_new, in_place=True)

    net.drop7 = L.Dropout(net.relu7, in_place=True)
    net.final = L.InnerProduct(net.drop7, num_output=args.num_out, param=learned_param, weight_filler=fc_filler)

  elif args.expType == "radar":
    net.silence = L.Silence(net.data, ntop=0)
    net.fc7_new = L.InnerProduct(net.radar, num_output=1024, param=learned_param, weight_filler=fc_filler)
    net.relu7 = L.ReLU(net.fc7_new, in_place=True)
    net.drop7 = L.Dropout(net.relu7, in_place=True)
    net.final = L.InnerProduct(net.drop7, num_output=args.num_out, param=learned_param, weight_filler=fc_filler)

  net.loss = L.SoftmaxWithLoss(net.final, net.label)
  net.acc = L.Accuracy(net.final, net.label)
  return net.to_proto()

"""
Training Code
"""
def create_data_txt(videos, stage='train'):
  fileName = osp.join(args.outputDir,"%s.txt" % (stage))
  frameFiles = []; frameLabels = []
    
  for v in videos:  
    # saving frames from video
    outDir = osp.join(args.tmpdir, 'vid_%s' % str(v.id))
    if not osp.exists(outDir):
      os.mkdir(outDir)

    io_utils.save_video_frames(osp.join(args.storeDir,v.path), outDir)
    framePaths = [outDir + '/%s.jpg' % str(x) for x in range(1, v.frame_count + 1)] 
    
    annObj = NSAnnotation.objects.get(_id=v.annotation_id)
    loc, glob = annObj.uncompress()
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
          frameFiles.append(framePaths[k])
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


def train_val_split(args, videos, trainRatio=0.8):
  n = len(videos)
  trainInd = np.random.choice(n, int(trainRatio*n), replace=False)
  valInd  = np.delete(np.arange(n), trainInd)

  trainVid = [videos[x] for x in trainInd]
  valVid = [videos[x] for x in valInd]

  trainIds = [str(v.id) for v in trainVid]
  valIds = [str(v.id) for v in valVid]
  return trainVid, trainIds, valVid, valIds

def setup(args):
  if not os.path.exists(args.outputDir):
    os.mkdir(args.outputDir)

  args.storeDir = "/mnt/Ext/data/"
  args.mean = [117.193,  117.673,  114.125]
  args.num_out = 4
  args.trainNet = osp.join(os.getcwd(), args.outputDir + 'train_net.prototxt')
  args.testNet  = osp.join(os.getcwd(), args.outputDir + 'test_net.prototxt')
  args.solverFile = osp.join(os.getcwd(), args.outputDir + 'solver.prototxt')
  args.tmpdir = get_basic_paths()['tmpvideo']['dr']

  if args.gpu >= 0:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)
  else:
    caffe.set_mode_cpu()


def main(args):
  videos = NSVideo.objects.filter(sensor_id__startswith='Nokia').filter(annotation_id__isnull=False)
  filteredVideos = []

  for v in videos:
    annObj = NSAnnotation.objects.get(_id=v.annotation_id)
    loc,glob = annObj.uncompress()
    if loc and ('101656' in v.path):
       filteredVideos.append(v)
  
  radarFiles = [osp.join(args.radarDir,'image3d_2017.01.12_10.%s.mat' % (str(int(v.path.split('.')[-2]) + 17))) for v in filteredVideos]
  for r,v in zip(radarFiles, filteredVideos):   
    if not osp.exists(r):
      radarFiles.remove(r)
      filteredVideos.remove(v)

  setup(args)
  #trainVideos, trainIds, valVideos, valIds = train_val_split(args, filteredVideos)

  #trainTxt = create_data_txt(trainVideos, stage="train")
  #valTxt = create_data_txt(valVideos, stage="val")
  trainTxt = osp.join(os.getcwd(), "runs_full/fullJoint/train.txt")
  valTxt = osp.join(os.getcwd(), 'runs_full/fullJoint/val.txt')
  trainIds = [x.rstrip().split()[0].split('/')[-2].split('_')[1] for x in open(trainTxt).readlines()]
  valIds = [x.rstrip().split()[0].split('/')[-2].split('_')[1] for x in open(valTxt).readlines()]

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
  parser.add_argument('--radarDir', type=str, default='/mnt/HardDrive/common/nokia_radar/sleeplab/fixedradar/', help='dir that contains radar files')
  parser.add_argument('--logFile', type=str, default='logs_full/fullRadar.log')
  parser.add_argument('--expType', nargs='?', choices=['radar','image','joint'], default='radar')
  parser.add_argument('--outputDir', type=str,default='runs_full/fullRadar/', help='dir to store random processing output')
  parser.add_argument('--weights', type=str, default='/mnt/HardDrive/common/caffe_models/bvlc_reference_caffenet.caffemodel', help="pretrained weights for loading")
  parser.add_argument('--snapshotDir',type=str, default='snapshots_full/fullRadar/', help='where to store training snapshots')
  parser.add_argument('--batchSize', type=int , default=32, help='batch size for training')
  parser.add_argument('--gpu', type=int, default=1, help='GPU used to train network; set to -1 for CPU training')
  return parser.parse_args()

if __name__=="__main__":
  args = parseArgs()
  main(args)

