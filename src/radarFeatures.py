"""Calculate features for fall detection from radar based on Su et al.
"Dopper Radar Fall Activity Detection Using the Wavelet Transform" """

from math import ceil
import scipy.io as sio
import numpy as np
import os.path as osp
import ipdb
import pywt
from scipy.signal import hamming

from radarUtils import file2dateTime

class RadarExtractor:
    """class for extracting radar features"""
    
    def __init__(self, dataFiles, frameRate=10):
        """pass in one of more data files for future feature extraction
           if more than one file is passed, they are assumed to be in contiguous order
           :param: dataFiles: must be in a tuple and 
                   is assumed to be in format image3d_2017.01.12_10.21.mat
        """
        # init frame rate, start time, and data
        self.frameRate = frameRate
        self.dataFiles = tuple(dataFiles)
        self.startTime = file2dateTime(self.dataFiles[0])   
        imsFull = None
        for dataFile in self.dataFiles:
            matContents = sio.loadmat(dataFile)
            ims = matContents['images']
            if imsFull is not None:
                imsFull = np.hstack((imsFull, ims))
            else:
                imsFull = ims
        self.data = imsFull[0]

    def extract_window_features(self, windowStart=None, windowStop=None, frameRate=10):
        """extract features from self.data from windowStart to windowStop
           :param: windowStart: time of start or none if start from data beginning
           :param: windowStop: time of stop or none if stop at data end
        """
        # calculate all energy values within window
        # TODO: properly handle windowing
        # TODO: properly handle advancing .25 seconds for each window
        dataWindow = self.data
        self.calc_energy(dataWindow,1,1,frameRate)
        for idx in range(6):
            i = idx+1
            energyWindow = []
            for j, im in enumerate(dataWindow):
                energyWindow.append(calc_energy(i, j))
    
        # normalize each energy value
    
    
        # create feature vector
    
    @staticmethod
    def waveletTransform(inVal):
        pass
    
    @staticmethod
    def calc_energy(data, i, j, frameRate):
        """calculate the energy from equation 4
           :param: i: index of WT coefficients at 2^i 
           :param: j: frame
        """
        # window size corresponding to 0.5s
        N = frameRate*0.5
        # energy calculation for this window
        out = 0 
        hamWin = hamming(N)
        # to replicate paper, calculate average velocity in the y direction
        ipdb.set_trace()
        ycurr = np.mean(data[0], (0,2))
        ydiffarr = np.zeros(ycurr.shape) # contains total movement vector
        yvel = np.zeros(len(data)) # contains approximation of average velocity form movement vector
        for i in range(len(data)):
            # convolve with derivative of gaussian kernel [-1, 0, 1], 0 padding
            if i<len(data)-1:
                ynext = np.mean(data[i+1], (0,2))
                if i>=1:
                    ydiff = ynext - yprev
                    ydiffarr = np.vstack((ydiffarr, ydiff))
                    yvel[i] = np.mean(np.absolute(ydiff))
                yprev = ycurr
                ycurr = ynext
            else:
                ydiffarr = np.vstack((ydiffarr, np.zeros(ycurr.shape)))

        for l in range(N):
            newVal = (hamWin[l]*waveletTransform(l+j(N/2)))**2
            out = out + newVal
        return out 
    
    @staticmethod
    def normalize_energy(frameRate):
        """normalize energy from equation 5 over a 2.5s period"""
        numFrames=int(ceil(2.5*frameRate))


# testing code
if __name__ == '__main__':
    # assume data is last 2 .mat files
    baseDir = '/mnt/HardDrive/common/nokia_radar/sleeplab'
    dataFiles = ('image3d_2017.01.12_10.29.mat', 'image3d_2017.01.12_10.30.mat') 
    radext = RadarExtractor([osp.join(baseDir, dataFile) for dataFile in dataFiles])
    # assume want to extract features from all data
    radext.extract_window_features()
