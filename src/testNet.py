import caffe
from caffe import layers as L
from caffe import params as P
import argparse

def test(args):
  if str(args.gpu) >= 0:
    caffe.set_mode_gpu()
    caffe.set_device(int(args.gpu))
  else:
    caffe.set_mode_cpu()
  
  net = caffe.Net(args.netFile, args.weights, caffe.TEST)
  avgAcc = 0.
  for i in range(args.iters):
    out = net.forward()
    print out['acc']
    avgAcc +=  out['acc']

  avgAcc /= float(args.iters)
  with open("output.txt",'w') as f:
    f.write('Average Acc: %s\n' % (str(avgAcc)))
  return

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--netFile', type=str, default='/home/vash/code/radar/src/output/test_net.prototxt')
  parser.add_argument('--weights', type=str, default='/home/vash/code/radar/src/snapshots/_iter_50000.caffemodel')
  parser.add_argument('--iters', type=int, default=316)
  parser.add_argument('--gpu', type=str, default='0')
  return parser.parse_args()

if __name__=="__main__":
  args = parse_args()
  test(args)
