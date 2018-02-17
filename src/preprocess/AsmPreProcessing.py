from sys import argv
import numpy as np
import re
import itertools
import scipy.misc
import os
import array
from scipy.misc import imsave

class asmprocessor(object):
    '''
    Generate features from ASM files
    '''
    # helper methods
    def asmToPng(self, asmFile, loc): # djaihfaig.asm
        f = open(asmFile)
        ln = os.path.getsize(asmFile) # orginal asm file size
        witdth = int(ln**0.5) # define a sqrt(length) as its witdth
        rem = ln%witdth # reminder
        a = array.array("B") #
        a.fromfile(f, ln-rem)

        g = np.reshape(a, (len(a)/witdth, witdth))
        g = np.uint8(g)
        imsave(loc + asmFile.split('/')[-1].split('.')[0] + '.png', g)

    # convert image to list
    def readImage(self, filename):
        '''
        convert image to a list, keep first [numThreshold] pixels
        '''
        f = open(filename,'rb')
        ln = os.path.getsize(filename) # length of file in bytes
        width = 256
        rem = ln%width
        a = array.array("B") # uint8 array
        a.fromfile(f,ln-rem)
        f.close()
        g = np.reshape(a,(len(a)/width,width))
        g = np.uint8(g)
        g.resize((numThreshold,))
        return [filename.split('/')[-1].split('.')[0]] + list(g)

if __name__ == '__main__':

    if len(argv) != 4:
        print"""
        Usage: python %s [asm_data_path] [png_data_path] [numThreshold]
        """ % argv[0]
        exit(1)

    asmprc = asmprocessor()
    path = argv[1]
    pngpath = argv[2]
    numThreshold=int(argv[3])

    for filename in os.listdir(path):
        asmprc.asmToPng(path + '/' + filename, pngpath + '/')
    
    result = []
    for pngfile in os.listdir(pngpath):
        result.append(asmprc.readImage(pngpath + '/' + pngfile))

    f = open('asmfeature.txt', 'w')
    for i in range(len(result)):
        f.write(str(result[i])+'\n')
    
    f.close()