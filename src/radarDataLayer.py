import os,glob
import caffe
import numpy as np
import cv2
import sys
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
    self.videos = params['videos']
    self.batchSize = int(params['batch_size'])
    self.radExt = RadarExtractor(self.radarFiles)    
    self.radarFeat = []
    self.idx = 0
    
    for v in self.videos:
      frameCount = v.frame_count
      start = v.start_timestamp
      spf = 1/(v.fps)
      for x in range(arameCount):
      	feat = radExt.extract_features(datetime.datetime.strptime(str(start + timedelta(seconds = spf*x), '%H:%M:%S'))     
        self.radarFeat.append(feat.flatten())
    
    self.radarFeat = np.array(self.radarFeat)
    self.numFeatures = len(self.radarFeat)
   
  def reshape(self, bottom, top):
    self.data  = self.load_radar_batch()
    top[0].reshape(*self.image.shape)

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
