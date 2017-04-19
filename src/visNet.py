import caffe
from caffe import layers as L
from caffe import params as P
import argparse
import os
import numpy as np

def softmax(vec):
  results = []
  for i in range(vec.shape[0]):
      results.append(np.argmax(np.exp(vec[i])))
  return np.array(results)

def vis(args):
  if args.gpu >= 0:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)
  else:
    caffe.set_mode_cpu()
  
  images = [x.rstrip().split()[0] for x in open(args.dataFile).readlines()]

  net = caffe.Net(args.netFile, args.weights, caffe.TEST)
  labels = []; pred = [];
  for i in range(args.iters):
    out = net.forward()
    labels = np.hstack((labels, net.blobs['label'].data))
    pred = np.hstack((pred, softmax(net.blobs['final'].data)))
   
  images = np.array(images[:args.iters*32])
  labels = labels[:args.iters*32]
  pred = pred[:args.iters*32]

  fp = images[(labels - pred) == -1] 
  fn = images[(labels - pred) == 1]
  if not os.path.exists('fp/'):
    os.mkdir('fp/')
  if not os.path.exists('fn/'):
    os.mkdir('fn/')
  
  for f in fp:
    os.system('cp %s %s' %(f,'fp/'))
  for g in fn:
    os.system('cp %s %s' %(g,'fn/'))
  return  

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataFile", type=str, default='runs/fullJoint/val.txt')
  parser.add_argument('--netFile', type=str, default='runs/fullJoint/test_net.prototxt')
  parser.add_argument('--weights', type=str, default='snapshots/fullJoint/_iter_20000.caffemodel')
  parser.add_argument('--iters', type=int, default=30)
  parser.add_argument('--gpu', type=int, default=1)
  return parser.parse_args()

if __name__=="__main__":
  args = parse_args()
  vis(args)
