# COPY OF ffmpeg.py from chainer repo
import os
import shutil
import random
from PIL import Image

class extract(object):
    def __init__(self, path, fps = None, size = None):
        self.output = "tmp"
        self.path = path
        try:
            os.makedirs(self.output)
        except:
            print 'Could not make dir {0}'.format(self.output)

        cmd = "ffmpeg -i {0} -b 10000k".format(path)
        if fps:
            cmd = "{0} -r {1}".format(cmd, int(fps))
        if size:
            w, h = size
            cmd = "{0} -s {1}x{2}".format(cmd, int(w), int(h))
        os.system("{0} {1}/%d.jpg".format(cmd, self.output))

    def __del__(self):
        if self.output:
            shutil.rmtree(self.output)

    def __getitem__(self, k):
        return Image.open(self.getframepath(k))

    def getframepath(self, k):
        return "{0}/{1}.jpg".format(self.output, k+1)

    def __len__(self):
        f = 1
        while True:
            if not os.path.exists(self.getframepath(f)):
                return f
            f += 1

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
