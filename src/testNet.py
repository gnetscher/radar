import caffe
from caffe import layers as L
from caffe import params as P
import argparse
import numpy as np

def test(args):
  if args.gpu >= 0:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)
  else:
    caffe.set_mode_cpu()
  
  net = caffe.Net(args.netFile, args.weights, caffe.TEST)
  per_class_total = [0,0,0,0]
  per_class_correct = [0,0,0,0]

  avgAcc = 0.
  for i in range(args.iters):
    out = net.forward()
    labels = net.blobs['label'].data
    pred = np.argmax(net.blobs['final'].data, axis=1)
    for i,l in enumerate(labels):
      per_class_total[int(l)] += 1
      if pred[i] == l:
        per_class_correct[int(l)] += 1    
    avgAcc +=  out['acc']


  avgAcc /= float(args.iters)
  with open(args.outFile,'w') as f:
    for q in range(len(per_class_total)):
      f.write("Class %s Acc: %s\n" %(str(q), str(float(per_class_correct[q])/per_class_total[q])))
    f.write('Average Acc: %s\n' % (str(avgAcc)))
  return

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--netFile', type=str, default='runs_full/fullRadar/test_net.prototxt')
  parser.add_argument('--weights', type=str, default='snapshots_full/fullRadar/_iter_20000.caffemodel')
  parser.add_argument('--iters', type=int, default=79)
  parser.add_argument('--outFile', type=str, default="results_full/fullRadar.txt")
  parser.add_argument('--gpu', type=int, default=1)
  return parser.parse_args()

if __name__=="__main__":
  args = parse_args()
  test(args)
