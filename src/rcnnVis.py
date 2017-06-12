from easydict import EasyDict as edict
import ipdb
import numpy as np
import os
import shutil
import sys
import cv2

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
from standalone import *

def get_videos():
  videos = NSVideo.fetch_annotated_videos(falls_only=False, home='Nokia')
  filteredVideos = []
  dataPath = get_basic_paths()['data']['dr'] + '/'
  for v in videos:
    v.download(dst_dir=dataPath, filename='%s.mp4' % str(v.id), overwrite=False)
    if v.annotations and v.start.month == 1:
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
    a = video.annotations
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
  prms = set_rcnn_prms(trainDataSet='coco', netName='vgg16-coco-rcnn')
  prms['targetClasses'] = ['person']
  prms['gpuId'] = 1
  rcnn = caffe_chains.Im2RCNNDet(prms)
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
    totalImageFile.append(imageFiles)
  return totalDet, totalImageFile


def save_detection(box, imageFile, outputDir):
    if not os.path.exists(outputDir):
      os.mkdir(outputDir)
    fName = imageFile.split('/')[-4] + "_" + imageFile.split('/')[-1]
    img = cv2.imread(imageFile)
    if len(box) != 0:
      x1, y1, x2, y2 = box[0][2]
      x, y = int(x1), int(y1)
      xh, yh = int(x2), int(y2)
      cv2.rectangle(img, (x,y), (xh, yh), (255,0,0),  2)
    cv2.imwrite(outputDir + fName, img)    

def get_ground_truth(videos):
  groundTruth = []
  for video in videos:
    loc = video.annotations
    act = jpkg_chains.Labels2MetricList()
    chain = Chainer([act])
    groundTruth.append(chain.produce(loc))
  return groundTruth

def visualize_results(totalDet, totalImageFile, totalGt):
  for det,images in zip(totalDet, totalImageFile):
    for k in range(len(det)):
      save_detection(det[k], images[k], "vis_output/")
  print "Done saving image, check vis_output/ for results"
    

if __name__=="__main__":
  session.authenticate('Nokia', '***********')
  videos = get_videos()
  fileList, paths = create_train_txt(videos)
  totalDet, totalIm = get_rcnn_detections(fileList)
  totalGt = get_ground_truth(videos)
  visualize_results(totalDet, totalIm, totalGt)
   
