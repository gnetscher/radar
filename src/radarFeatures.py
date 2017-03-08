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
     
    @staticmethod
    def wavelet_transform(inVal):
        return pywt.swt(inVal, 'rbio3.3')

    def __init__(self, dataFiles, frameRate=10, scaleFactor=4e-6):
        """pass in one of more data files for future feature extraction
           if more than one file is passed, they are assumed to be in contiguous order
           :param: dataFiles: must be in a tuple and 
                   is assumed to be in format image3d_2017.01.12_10.21.mat
           :param: frameRate: sampling rate of the radar system in Hz
           :param: scaleFactor: factor used to increase scale to avoid roundoff errors
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
        self.data = imsFull[0]/scaleFactor
        self.yvel = self.calc_avg_vel(self.data)
        self.yvelWaveCo = np.array(self.wavelet_transform(self.yvel))

    # TODO: true positive should only be the period of 2.5s around the transition from
    #       any position to on the ground
    def extract_window_features(self, windowStart=None, windowStop=None, frameRate=10):
        """extract features from self.data from windowStart to windowStop
           :param: windowStart: time of start or none if start from data beginning
           :param: windowStop: time of stop or none if stop at data end
        """
        # calculate all energy values within window
        # TODO: properly handle windowing
        # TODO: properly handle advancing .25 seconds for each window (M=7 b/c window=1s)
        dataWindow = self.data
        self.calc_energy(1,1,frameRate)
        for i in range(yvelWaveCo.shape[0]*yvelWaveCo.shape[1]):
            energyWindow = []
            for j, im in enumerate(dataWindow):
                energyWindow.append(calc_energy(i, j))
    
        # normalize each energy value
    
    
        # create feature vector
   
    def calc_avg_vel(self, data):
        # to replicate paper, calculate average velocity in the y direction
        ycurr = np.mean(data[0], (0,2))
        ydiffarr = np.zeros(ycurr.shape) # contains total movement vector
        yvel = np.zeros(len(data)) # contains approximation of average velocity form movement vector
        for idx in range(len(data)):
            # convolve with derivative of gaussian kernel [-1, 0, 1], 0 padding
            if idx<len(data)-1:
                ynext = np.mean(data[idx+1], (0,2))
                if idx>=1:
                    ydiff = ynext - yprev
                    ydiffarr = np.vstack((ydiffarr, ydiff))
                    yvel[idx] = np.mean(np.absolute(ydiff))
                yprev = ycurr
                ycurr = ynext
            else:
                ydiffarr = np.vstack((ydiffarr, np.zeros(ycurr.shape)))
        return yvel
   
    def calc_energy(self, i, j, frameRate):
        """calculate the energy from equation 4
           :param: i: index of WT coefficients at 2^i 
           :param: j: frame (note a frame is composed of N samples -- there are len(data)/N frames)
        """
        # window size corresponding to 1s (note paper says 0.5, but our frame rate is lower,
        # so I don't want to make the window too small)
        N = int(frameRate*1)
        # energy calculation for this window
        out = 0 
        hamWin = hamming(N)
        # calculate energy from eq 4
        ipdb.set_trace()
        for l in range(N):
            newVal = (hamWin[l]*self.yvelWaveCo[i/2][i%2][l+j*N/2])**2
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
