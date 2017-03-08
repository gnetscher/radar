"""Calculate features for fall detection from radar based on Su et al.
"Dopper Radar Fall Activity Detection Using the Wavelet Transform" """

import scipy.io as sio
import numpy as np
import os.path as osp
import ipdb
import pywt
from scipy.signal import hamming
import datetime

from radarUtils import file2dateTime

class RadarExtractor:
    """class for extracting radar features
       Note: the majority of work is completed in the constructor as preprocessing,
       so calls to extract_features occur quickly without repeating work
    """
     
    @staticmethod
    def wavelet_transform(inVal):
        # use stationary wavelet transform with rbio3.3 kernel as recommended in paper
        return pywt.swt(inVal, 'rbio3.3')

    def calc_frame_energies(self):
        """calculate the energy in self.data for frames composed of 10 samples with stride 2
           then normalize over 2.6s period (eq 5) roughly corresponding to fall length
        """
        # calc energies - eq 4
        numWTLevels = self.yvelWaveCo.shape[0]*self.yvelWaveCo.shape[1]
        energies = np.zeros((numWTLevels, self.numFrames))
        for i in range(numWTLevels):
            for j in range(self.numFrames):
                energies[i, j] = self.calc_energy(i, j)
        # normalize - eq 5
        normedEnergies = np.zeros((numWTLevels, self.numFrames))
        for i in range(numWTLevels):
            for j in range(self.numFrames):
                start = max(j-self.M, 0)
                stop  = min(j+self.M+1, self.numFrames)
                if stop - start == 2*self.M+1:
                    dividend = sum(energies[i, start:stop])
                else:
                    # use edge value for normalizing at edges
                    if start == 0:
                        dividend = sum(energies[i, start:stop]) + \
                                      (2*self.M+1 - (stop - start))*energies[i, start]
                    else:
                        dividend = sum(energies[i, start:stop]) + \
                                      (2*self.M+1 - (stop - start))*energies[i, stop-1]
                normedEnergies[i, j] = energies[i, j] / dividend  
        return normedEnergies        

    def __init__(self, dataFiles):
        """pass in one of more data files for future feature extraction
           if more than one file is passed, they are assumed to be in contiguous order
           :param: dataFiles: must be in a tuple and 
                   is assumed to be in format image3d_2017.01.12_10.21.mat
        """
        # init frame rate, start time, and data
        self.frameRate = 10 
        self.frameLen  = 10
        self.shiftLen  = 2
        self.fallLen   = 26
        self.scaler    = 4e-6
        self.dataFiles = tuple(dataFiles)
        self.startTime = file2dateTime(self.dataFiles[0])   
        self.endTime   = self.startTime + datetime.timedelta(minutes=len(dataFiles))
        imsFull = None
        for dataFile in self.dataFiles:
            matContents = sio.loadmat(dataFile)
            ims = matContents['images']
            if imsFull is not None:
                imsFull = np.hstack((imsFull, ims))
            else:
                imsFull = ims
        self.data = imsFull[0]/self.scaler
        self.yvel = self.calc_avg_vel(self.data)
        self.yvelWaveCo = np.array(self.wavelet_transform(self.yvel))
        self.M         = ((self.fallLen - self.frameLen) / self.shiftLen + 1) / 2
        self.numFrames = (self.yvelWaveCo.shape[2] - self.frameLen) / (self.frameLen / self.shiftLen) + 1
        self.frameEnergies = self.calc_frame_energies()

    def extract_features(self, time):
        """extract features from self.data at the frame corresponding to time 
           :param: time: datetime object corresponding to exact time wanting features extracted
        """
        # find matching frame
        
        timeRatio = (float((3600*time.hour + 60*time.minute + time.second)) - \
                     (3600*self.startTime.hour + 60*self.startTime.minute + self.startTime.second)) / \
                    ((3600*self.endTime.hour + 60*self.endTime.minute + self.endTime.second) - \
                     (3600*self.startTime.hour + 60*self.startTime.minute + self.startTime.second))
        if timeRatio < 0.0 or timeRatio > 1.0:
            raise ValueError('Time specified is %s which is not within data window' % time)
        frame = int(timeRatio*self.numFrames)

        ipdb.set_trace()
        # output feature vector
        start = max(frame-self.M, 0)
        stop  = min(frame+self.M+1, self.numFrames)
        if stop - start == 2*self.M+1:
            features = self.frameEnergies[:, start:stop]
        else:
            # use edge values to reach correct vector length
            if start == 0:     
                features = self.frameEnergies[:, start]
                for _ in range(1, 2*self.M+1 - (stop-start)):
                    features = np.vstack((features, self.frameEnergies[:, start]))
                features = np.transpose(features)
                features = np.hstack((features, self.frameEnergies[:, start:stop]))
            else:
                features = self.frameEnergies[:, start:stop]
                for _ in range(2*self.M+1 - (stop-start)):
                    features = np.hstack((features, self.frameEnergies[:, stop-1, None]))

        return features

   
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
   
    def calc_energy(self, i, j):
        """calculate the energy from equation 4
           :param: i: index of WT coefficients at 2^i 
           :param: j: frame (note a frame is composed of N samples -- there are len(data)/N frames)
        """
        # window size corresponding to 1s (note paper says 0.5, but our frame rate is lower,
        # so I don't want to make the window too small)
        N = self.frameLen
        # energy calculation for this window
        out = 0 
        hamWin = hamming(N)
        # calculate energy from eq 4
        for l in range(N):
            newVal = (hamWin[l]*self.yvelWaveCo[i/2][i%2][l+j*N/5])**2
            out = out + newVal
        return out 
    

# testing code
if __name__ == '__main__':
    # assume data is last 2 .mat files
    baseDir = '/mnt/HardDrive/common/nokia_radar/sleeplab'
    dataFiles = ('image3d_2017.01.12_10.29.mat', 'image3d_2017.01.12_10.30.mat') 
    radext = RadarExtractor([osp.join(baseDir, dataFile) for dataFile in dataFiles])
    # assume want to extract features from all data
    radext.extract_features(datetime.datetime.strptime('10:30:59', '%H:%M:%S'))
