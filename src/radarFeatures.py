"""Calculate features for fall detection from radar """

import scipy.io as sio
import numpy as np
import os.path as osp
import ipdb
import pywt
from scipy.signal import hamming, medfilt
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

    def __init__(self, dataFiles, featureType='velocity'):
        """pass in one of more data files for future feature extraction
           if more than one file is passed, they are assumed to be in contiguous order
           :param: dataFiles: must be in a tuple and 
                   is assumed to be in format image3d_2017.01.12_10.21.mat
           :param: featureType: 'velocity' or 'position' - specifies the type of feature vector to extract
        """
        # init constants
        self.frameRate = 10 
        self.frameLen  = 10
        self.shiftLen  = 2
        self.fallLen   = 26
        self.scaler    = 4e-6
        self.dataFiles = tuple(dataFiles)
        self.startTime = file2dateTime(self.dataFiles[0])   
        self.endTime   = self.startTime + datetime.timedelta(minutes=len(dataFiles))
        self.featType  = featureType
        # load data
        imsFull = None
        for dataFile in self.dataFiles:
            matContents = sio.loadmat(dataFile)
            ims = matContents['images']
            if imsFull is not None:
                imsFull = np.hstack((imsFull, ims))
            else:
                imsFull = ims
        self.data = imsFull[0]/self.scaler
        # perform preprocessing
        if featureType=='velocity':
            self.yvel = self.calc_avg_vel(self.data)
            self.yvelWaveCo = np.array(self.wavelet_transform(self.yvel))
            self.M         = ((self.fallLen - self.frameLen) / self.shiftLen + 1) / 2
            self.numFrames = (self.yvelWaveCo.shape[2] - self.frameLen) / (self.frameLen / self.shiftLen) + 1
            self.frameEnergies = self.calc_frame_energies()
        elif featureType=='position':
            pass

    def extract_velocity_features(self, time, stopWindow=None):
        """extract velocity features based on Su et al.
           "Dopper Radar Fall Activity Detection Using the Wavelet Transform" 
        """

        def time2frame(time):
            # find matching frame for time
            timeRatio = (float((3600*time.hour + 60*time.minute + time.second)) - \
                         (3600*self.startTime.hour + 60*self.startTime.minute + self.startTime.second)) / \
                        ((3600*self.endTime.hour + 60*self.endTime.minute + self.endTime.second) - \
                         (3600*self.startTime.hour + 60*self.startTime.minute + self.startTime.second))
            if timeRatio < 0.0 or timeRatio > 1.0:
                raise ValueError('Time specified is %s which is not within data window' % time)
            frame = int(timeRatio*self.numFrames)
            return frame
        
        # find matching frame for time and stopWindow
        startFrame = time2frame(time)
        if stopWindow is not None:
            stopFrame = time2frame(stopWindow)
        else:
            stopFrame = startFrame + 1

        # create feature vector
        featureVec = []
        for frame in range(startFrame, stopFrame):
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
            featureVec.append(features)

        # return list if window requested, return features for individual frame if only frame requested
        if stopWindow is not None:
            return featureVec
        else:
            return features

    def extract_position_features(self, time, stopWindow=None):
        """experiments with extracting more relevant features to indicate if someone is on the ground 
        """
        def time2sample(time):
            # find matching sample for time (note a sample is different from a frame)
            timeRatio = (float((3600*time.hour + 60*time.minute + time.second)) - \
                         (3600*self.startTime.hour + 60*self.startTime.minute + self.startTime.second)) / \
                        ((3600*self.endTime.hour + 60*self.endTime.minute + self.endTime.second) - \
                         (3600*self.startTime.hour + 60*self.startTime.minute + self.startTime.second))
            if timeRatio < 0.0 or timeRatio > 1.0:
                raise ValueError('Time specified is %s which is not within data window' % time)
            sample = int(timeRatio*len(self.data))
            return sample

        # find matching sample for time and stopWindow
        startSample = time2sample(time)
        if stopWindow is not None:
            stopSample = time2sample(stopWindow)
        else:
            stopSample = startSample + 1

        # create feature vector
        featureVec = []
        for sample in range(startSample, stopSample):
            features = self.calc_sample_features(self.data[sample])
            featureVec.append(features)

        # return list if window requested, return features for individual sample if only sample requested
        if stopWindow is not None:
            return featureVec
        else:
            return features

    def extract_features(self, time, stopWindow=None):
        """extract features from self.data at the frame corresponding to time based on featureType specified
           in constructor
           :param: time: datetime object corresponding to exact time wanting features extracted
           :param: stopWindow: allows for specifying range. if not None, a feature array will be 
                   returned for all frames between time and stopWindow
        """
        if self.featType == 'velocity':
            return self.extract_velocity_features(time, stopWindow)
        elif self.featType == 'position':
            return self.extract_position_features(time, stopWindow)
           
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

    def calc_sample_features(self, sampleData, threshold=2):
        """calculate the following 54 features for an individual sample, arrange as 6x9matrix to match velocity
           (0-7)   max eight values
           (8-31)  locations of max 8 values in theta,phi,z
           (32-34) variance of locations of max 8 values in (theta, phi, z)
           (35-37) range of max values in theta,phi,z
           (38-40) variance in theta,phi,z
           (41)    min value
           (42)    mean value
           (43)    max value after smoothing
           (44-46) index of max value after smoothing in theta,phi,z
           (47)    number greater than threshold
           (48)    variance of values above threshold 
           (49)    number greater than threshold x2
           (50)    variance of values above threshold x2
           (51-54) variance of indexes of values above threshold in theta,phi,z
           :param: sample: the index of the radar data to calculate features for
           :param: threshold: the threshold used to explore values above a certain noise floor
        """
        numFeatures=54
        features = np.zeros(numFeatures)
        im = abs(sampleData)
        # max values
        numMaxValues=8
        maxInd = np.argpartition(im, -numMaxValues, axis=None)[-numMaxValues:]
        maxValues = im.flatten()[maxInd]
        features[:numMaxValues] = maxValues
        # locations of max values
        for i, ind in enumerate(maxInd):
            z = ind / (im.shape[0] * im.shape[1]) 
            phi = (ind - z*im.shape[0]*im.shape[1]) / im.shape[0]
            theta = ind - (z*im.shape[0]*im.shape[1]) - (phi*im.shape[0])
            features[numMaxValues+i*3] = theta
            features[numMaxValues+i*3+1] = phi 
            features[numMaxValues+i*3+2] = z
        # variance of locations of max 8 values
        features[32] = np.var(features[8:32:3])
        features[33] = np.var(features[9:32:3])
        features[34] = np.var(features[10:32:3])
        # range of locations of max 8 values
        features[35] = np.ptp(features[8:32:3])
        features[36] = np.ptp(features[9:32:3])
        features[37] = np.ptp(features[10:32:3])
        # variance of all values in theta, phi, z
        features[38] = np.var(np.mean(im, axis=(1,2)))
        features[39] = np.var(np.mean(im, axis=(0,2)))
        features[40] = np.var(np.mean(im, axis=(0,1)))
        # min, mean
        features[41] = np.min(im)
        features[42] = np.mean(im)
        # smooth with median filter and get max, index of first max
        smoothIm = medfilt(im, 5)
        features[43] = np.max(smoothIm)
        smimIdx = np.where(smoothIm==features[43])
        features[44] = smimIdx[0][0]
        features[45] = smimIdx[1][0]
        features[46] = smimIdx[2][0]
        # threshold and find number and variance above threshold
        threshIdx = np.where(im>threshold)
        features[47] = len(threshIdx[0])
        features[48] = np.var(im[threshIdx])
        threshIdx2 = np.where(im>2*threshold)
        features[49] = len(threshIdx2[0])
        features[50] = np.var(im[threshIdx2])
        features[51] = np.var(threshIdx[0])
        features[52] = np.var(threshIdx[1])
        features[53] = np.var(threshIdx[2])
        # return final features vector
        return np.nan_to_num(features.reshape(6,9))
    
# testing code
if __name__ == '__main__':
    # assume data is last 2 .mat files
    baseDir = '/mnt/HardDrive/common/nokia_radar/sleeplab'
    dataFiles = ('image3d_2017.01.12_10.29.mat', 'image3d_2017.01.12_10.30.mat') 
    radext = RadarExtractor([osp.join(baseDir, dataFile) for dataFile in dataFiles], featureType='position')
    # assume want to extract features from all data
    startTime = datetime.datetime.strptime('10:29:30', '%H:%M:%S')
    endTime = datetime.datetime.strptime('10:29:35', '%H:%M:%S')
    feat = radext.extract_features(startTime, endTime)
    print feat
