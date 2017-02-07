"""functions for manipulating radar data"""

import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
import numpy as np
import ipdb
import datetime

def frames2movie(ims, outFile, frames):    
    ''' convert radar frames into movie - for internal use by plot_radar'''
    # set up figure
    fig = plt.figure(figsize=(8,6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,1])
    ax = plt.subplot(gs[0])
    ax.set_xlabel('Z (R)'); ax.set_ylabel(r'$X (\theta)$')

    ax2 = plt.subplot(gs[1])
    ax2.set_xlabel('Z (R)'); ax2.set_ylabel(r'$Y (\phi)$');

    plt.tight_layout()

    # functions to create single frames
    def make_frame(idx):
        tIm = ims[:, idx][0]
        fig.suptitle('Frame %d' % idx)
        ax2.imshow(np.squeeze(np.mean(abs(tIm), 0))/4e-6)
        ax.imshow(np.squeeze(np.mean(abs(tIm), 1))/4e-6)

    if frames is None:
        frameIter = range(ims.shape[1])
    else:
        frameIter = range(frames)
    ani = animation.FuncAnimation(fig, make_frame, frames=frameIter, blit=True, interval=10) 

    # make movie
    writer = animation.writers['ffmpeg'](fps=10)
    ani.save(outFile, writer=writer)

def plot_radar(inFile, outFile='out.mp4', frames=None):
    '''read in data file and produce movie'''
    matContents = sio.loadmat(inFile)
    ims = matContents['images']
    frames2movie(ims, outFile, frames)

def file2dateTime(inFile):
    """extract datetime object from filename in format image3d_2017.01.12_10.21.mat"""
    (im, cal, tim) = inFile.split('_')
    ctim = tim.split('.')
    dstr = cal + '.' + ctim[0] + '.' + ctim[1]
    return datetime.datetime.strptime(dstr, '%Y.%m.%d.%H.%M')  
