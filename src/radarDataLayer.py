import os,glob
import caffe
import numpy as np
import cv2
import sys
from ns_backend import *
from chainer.objects import data_objects as dj
import datetime
from datetime import timedelta
from radarFeatures import RadarExtractor
import random

class RadarDataLayer(caffe.Layer):
  """
  Loading image, label from the medical data files
  """	

  def setup(self, bottom, top):
    params = eval(self.param_str) 
    self.radarFiles = params['radar_files']
    self.videoIds = params['videos']
    self.videos = NSVideo.objects.filter(_id__in=self.videoIds)
    self.videos = [dj.Video(v) for v in self.videos]
    
    self.batchSize = int(params['batch_size'])
    self.radExt = RadarExtractor(self.radarFiles, featureType='position')    
    self.radarFeat = []
    self.idx = 0
    self.spf = 1/(self.videos[0].fps)    


    for v in self.videos:
      frameCount = v.frame_count
      start = v.start_timestamp
      #if start.minute < 17:
      #	start = datetime.datetime(start.year, start.month, start.day, start.hour, 17, 0)
      for x in range(frameCount):
        feat = self.radExt.extract_features(start + timedelta(seconds = self.spf*x))
        self.radarFeat.append(feat.flatten())
    
    self.radarFeat = np.array(self.radarFeat)
    self.numFeatures = len(self.radarFeat)
 
  def reshape(self, bottom, top):
    self.data  = self.load_radar_batch()
    top[0].reshape(*self.data.shape)

  def forward(self, bottom, top):
    top[0].data[...] = self.data
		
    self.idx += self.batchSize
    if self.idx > self.numFeatures:
      self.idx = self.idx % self.numFeatures

  def backward(self, top, propogate_down, bottom):
    pass

  def load_radar_batch(self):
    indices  = [(self.idx + i) % self.numFeatures for i in range(self.batchSize)]
    data_batch = self.radarFeat[indices].astype(np.float32)
    return data_batch
