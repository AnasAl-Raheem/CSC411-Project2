
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
import hashlib
from PIL import Image

# constants
actors_file = "facescrub_actors.txt"
actresses_file = "facescrub_actresses.txt"
actors_folder = "uncropped_male_rgb/"
actresses_folder = "uncropped_female_rgb/"
actors = list(set([a.split("\t")[0] for a in open(actors_file).readlines()]))
actresses = list(set([a.split("\t")[0] for a in open(actresses_file).readlines()]))

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

def get_sha256(filePath, block_size = 2**20):
    with open(filePath, 'rb') as file:
        hash = hashlib.sha256()
        while True:
            data = file.read(block_size)
            if not data:
                break
            hash.update(data)
        return hash.hexdigest()

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    if len(rgb.shape) == 3:
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray / 255.
    else:
        return rgb



#Note: you need to create the uncropped folder first in order 
#for this to work

def download_pictures(act_list, act_file, folder_name):
    testfile = urllib.URLopener()
    for a in act_list:
        name = a.split()[1].lower()
        i = 0
        for line in open(act_file):
            if a in line:
                line_split = line.split()
                filename = name+str(i)+'.'+line_split[4].split('.')[-1]
                #A version without timeout (uncomment in case you need to
                #unsupress exceptions, which timeout() does)
                # try:
                #     testfile.retrieve(line_split[4], folder_name + filename)
                # except IOError as e:
                #     print "I/O error({0}): {1}".format(e.errno, e.strerror)
                #timeout is used to stop downloading images which take too long to download
                timeout(testfile.retrieve, (line_split[4], folder_name + filename), {}, 45)
                if not os.path.isfile(folder_name + filename):
                    continue
                if get_sha256(folder_name + filename) != line_split[6] and (name != "gilpin" and name != "bracco"):
                    print "hash doesn't match for " + filename + " " + "link:" + line_split[4]
                    try:
                        os.remove(folder_name + filename)
                    except WindowsError as e:
                        print "WindowsError error({0}): {1}".format(e.errno, e.strerror)
                    continue

                try:
                    im = imread(folder_name + filename)
                    dim = line_split[5].split(",")
                    im = im[int(dim[1]):int(dim[3]), int(dim[0]):int(dim[2])]
                    if len(im.shape) != 3:
                        im = np.array([im.T, im.T, im.T], ndmin=3).T
                    imsave(folder_name[2:-1] + "227/" + filename, imresize(im, (227, 227)))
                    # imsave(folder_name[2:-1] + "64/" + filename, imresize(im, (64, 64)))
                    # imsave(folder_name[2:] + filename, imresize(rgb2gray(im), (32, 32)), cmap=plt.cm.gray)

                    print filename
                    i += 1
                except IOError as e:
                    try:
                        os.remove(folder_name + filename)
                    except WindowsError as e:
                        print "WindowsError error({0}): {1}".format(e.errno, e.strerror)
                    print "I/O error({0}): {1}".format(e.errno, e.strerror)
                except IndexError as e:
                    try:
                        os.remove(folder_name + filename)
                    except WindowsError as e:
                        print "WindowsError error({0}): {1}".format(e.errno, e.strerror)
                    print "IndexError e:", e
download_pictures(actors, actors_file, actors_folder)
download_pictures(actresses, actresses_file, actresses_folder)