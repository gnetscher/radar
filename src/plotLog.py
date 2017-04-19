import os
import matplotlib.pyplot as plt
import argparse


def main(args):
  lines = [x.rstrip() for x in open(args.log).readlines()]
  
  #training loss
  if args.option == 0:
    title = "Training Loss"
    iters = [l for l in lines if 'Iteration' in l and 'loss' in l]
    iters = [int(a.split(']')[1].split()[1].split(',')[0]) for a in iters]
    loss = [l for l in lines if 'Train net' in l and 'loss' in l]
    loss = [float(a.split('=')[-2].split('(')[0]) for a in loss]

  #validation loss
  elif args.option == 1:
    title = "Validation Loss"
    iters = [l for l in lines if 'Iteration' in l and 'Testing' in l]    
    iters = [int(a.split(']')[1].split()[1].split(',')[0]) for a in iters]
    loss = [l for l in lines if 'Test net' in l and 'loss' in l]
    loss = [float(a.split('=')[-2].split('(')[0]) for a in loss]

  # training accuracy
  elif args.option == 2:
    title = "Training Accuracy"
    iters = [l for l in lines if 'Iteration' in l and 'loss' in l]
    iters = [int(a.split(']')[1].split()[1].split(',')[0]) for a in iters]
    loss = [l for l in lines if 'Train net' in l and 'acc' in l]
    loss = [float(a.split('=')[-1]) for a in loss]

  # validation accuracy
  elif args.option == 3:
    title = "Validation Accuracy"
    iters = [l for l in lines if 'Iteration' in l and 'Testing' in l]
    iters = [int(a.split(']')[1].split()[1].split(',')[0]) for a in iters]
    loss = [l for l in lines if 'Test net' in l and 'acc' in l]
    loss = [float(a.split('=')[-1]) for a in loss]
  
  else:
    print "Please enter a valid plotting option 0-3..."
    return 
  
  plt.title(title)
  plt.xlabel("Iteration")
  if args.option == 0 or args.option == 1:
    plt.ylabel("Loss")
  elif args.option == 2 or args.option == 3:
    plt.ylabel("Accuracy")
  
  plt.plot(iters,loss)
  plt.show()

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--log', type=str, default='logs/halfJoint.log', help='training log')
  parser.add_argument('--option', type=int, default=0, help="0 for training and 1 for validation")
  return parser.parse_args()


if __name__=="__main__":
  args = parse_args()
  main(args)
