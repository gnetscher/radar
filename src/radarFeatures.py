"""Calculate features for fall detection from radar based on Su et al.
"Dopper Radar Fall Activity Detection Using the Wavelet Transform" """

from math import ceil
import scipy.io as sio
import numpy as np
import os.path as osp
import ipdb

from radarUtils import file2dateTime

class RadarExtractor:
    """class for extracting radar features"""
    
    def __init__(self, dataFiles, frameRate=10):
        """pass in one of more data files for future feature extraction
           if more than one file is passed, they are assumed to be in contiguous order
           :param: dataFiles must be in a tuple and 
                   is assumed to be in format image3d_2017.01.12_10.21.mat"""
        # init frame rate, start time, and data
        self.frameRate = frameRate
        self.dataFiles = tuple(dataFiles)
        ipdb.set_trace()
        self.startTime = file2dateTime(self.dataFiles[0])   
        imsFull = None
        for dataFile in self.dataFiles:
            matContents = sio.loadmat(dataFile)
            ims = matContents['images']
            if imsFull is not None:
                imsFull = np.hstack((imsFull, ims))
            else:
                imsFull = ims
        self.data = imsFull

    def extract_window_features(self, windowStart=None, windowStop=None):
        """extract features from self.data from startTime to stopTime"""
        # calculate all energy values within window
         
    
        # normalize each energy value
    
    
        # create feature vector
    
    @staticmethod
    def waveletTransform(inVal):
        pass
    
    @staticmethod
    def calc_energy(i, j, N=480):
        """calculate the energy from equation 4"""
        out = 0 
        for l in range(480):
            #hamWin = ??
            newVal = (hamWin*waveletTransform(l+j(N/2)))^2
            out = out + newVal
        return out 
    
    @staticmethod
    def normalize_energy(frameRate):
        """normalize energy from equation 5 over a 2.5s period"""
        numFrames=int(ceil(2.5*frameRate))


# testing code
if __name__ == '__main__':
    baseDir = '/home/nestsense/code/nokia/data'
    dataFiles = ('image3d_2017.01.12_10.29.mat', 'image3d_2017.01.12_10.30.mat') 
    RadarExtractor([osp.join(baseDir, dataFile) for dataFile in dataFiles])
