from easydict import EasyDict as edict
import numpy as np
import os
import shutil
import sys
# self imports
from chainer.objects import experiment_objects as eo
from chainer.objects import caffe_experiment_objects as ceo
from chainer.objects import experiment_session
from chainer.chains.chain import Chainer
from chainer.chains import metric_chains
from chainer.chains import jpkg_chains
from chainer.chains import image_chains
from chainer.chains import caffe_chains
from chainer.config import *
from standalone import session, NSVideo, NSEnum

def get_videos():
  videos = NSVideo.objects.filter(sensor_id__startswith='Nokia').filter(annotation_id__isnull=False)
  filteredVideos = []
  dataPath = get_basic_paths()['data']['dr'] + '/'
  for v in videos:
    v.download(dst_dir=dataPath)
    if v.annotations and v.annotations[1] and ('101656' in v.path):
       filteredVideos.append(v)
  return filteredVideos

def create_train_txt(videos):
  fileList = []
  framePath =  get_basic_paths()['tmpvideo']['dr'] + '/'
  dataPath = get_basic_paths()['data']['dr'] + '/'
  trainDataSet = 'coco'
  paths = []
  outputDir = "tempdata/"
  if not osp.exists(outputDir):
    os.mkdir(outputDir)

  for video in videos:
    p = video.path
    a = video.annotations[1]
    w = video.width
    h = video.height
    labelObj = jpkg_chains.Labels2RCNNTrainTxt({'outFile': '{0}train{1}.txt'.format(
               outputDir, video.id), 'imageFolder': framePath,
               'category': 'object', 'trainDataSet': trainDataSet})
    outFile = labelObj.produce([{'vidName': p, 'labels': a,
              'height': h, 'width': w}])
    fileList.append(outFile)
    savePath = framePath + os.path.basename(p).split('.')[0] + '/'
    paths.append(savePath)
  return fileList, paths

def get_rcnn_detections(fileList):
  imProd = image_chains.File2Im()
  bgr = image_chains.RGB2BGR()
  rcnn = caffe_chains.Im2RCNNDet()
  jlbl = jpkg_chains.RCNN2Labels()

  totalDet = []
  totalImageFile = []
  for f in fileList:
    imageFiles = list(set([x.split()[0] for x in open(f).readlines()]))
    imageFiles = sorted(imageFiles,
                 key=lambda x: int(x.split('/')[-1].split('.')[0]))
    chain = Chainer([imProd, bgr, rcnn])
    pred = jpkg_chains.Labels2MetricList()
    chain2 = Chainer([jlbl, pred])
    allDet = []
    for im in imageFiles:
      chainOut = chain.produce(im)
      allDet.append(chainOut)
    chain2Out = chain2.produce(allDet)
    totalDet.append(chain2Out)
    totalImageFile += imageFiles
  return totalDet, totalImageFile


def create_patches(boxes, im):
    for b in boxes:
      x1, y1, x2, y2 = b[1]
      x, y = np.floor(x1), np.floor(y1)
      xh, yh = np.floor(x2), np.floor(y2)
      cv2.rectangle(im, (x,y), (xh, yh), 'r',  2)
    
def get_ground_truth(videos):
  groundTruth = []
  for video in videos:
    loc = video.annotations[1]
    act = jpkg_chains.Labels2MetricList()
    chain = Chainer([act])
    groundTruth.append(chain.produce(loc))
  return groundTruth

def visualize_results(totalDet, totalImageFile, totalGt):
  resultDir = "vis_output/"
  for det,im in zip(totalDet, totalImageFile):
    print "Image: " + str(im)
    print "Pred: " + str(det)
    fname = im.split('/')[-1]
    img = cv2.imread(im)
    create_patches(det, img)
    cv2.imwrite(resultDir + fname, img)
  print "Done saving image, check %s for results" % resultDir
    

if __name__=="__main__":
  session.authenticate('Nokia', 'nokia')
  videos = get_videos()
  fileList, paths = create_train_txt(videos)
  totalDet, totalIm = get_rcnn_detections(fileList)
  totalGt = get_ground_truth(videos)
  visualize_results(totalDet, totalIm, totalGt)
   
