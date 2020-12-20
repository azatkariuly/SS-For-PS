import torch
import torch.nn as nn
import torchvision.transforms as transforms
import math
import numpy as np
from .binarized_modules import  BinarizeLinear,BinarizeConv2d,custom_quantize, Binarize
import matplotlib.pyplot as plt
import scipy.stats as stats

__all__ = ['resnet_binary']

quantize = True
graph = False
binarize = False

#global partial_sums

#partial_sums = []
##################################  PARTIAL SUMS PARAMETERS CONTROL #################################

#quantize = True
#binarize = True
#graph = True

scale = 1
thresh = 8000

num_bit = 1


############################################################################

kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=100)

file = str(num_bit) + '_bit.png'



def Binaryconv1x1(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return BinarizeConv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False, nbits=num_bit, quantize=quantize)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=1, bias=False)

def plot_dist(tensor):
    plt.hist(tensor, **kwargs)
    plt.title('histogram for partial sums')
    plt.savefig(file)

import collections
def freq_mapping(tensor):
    #print(tensor)
    out = {}
    for word in tensor:
        if word not in out:
            out[word] = 1
        else:
            out[word] += 1
    return out


def plot_importance(tensor):
    tensor = np.sort(np.absolute(tensor))
    freq =  freq_mapping(tensor)
    imp = []
    mag = []
    #print(freq)
    for key in freq:
        imp.append(key*freq[key])
        mag.append(key)
    #print(mag)
    #print(imp)
    plt.plot(mag,imp)
    plt.title('histogram for importance of partial sums')
    plt.savefig('importance_map.png')

def init_model(model):
    for m in model.modules():
        if isinstance(m, BinarizeConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

def split_tensor_128(xp):
    x1 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2)
    x2 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2)
    x3 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2)
    x4 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2)
    x5 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2)
    x6 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2)
    x7 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2)
    x8 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2)
    x9 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2)

    return x1,x2,x3,x4,x5,x6,x7,x8,x9

def split_tesnsor_256(xp,max_size = 128):
    x1 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,0,max_size)
    x2 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,0,max_size)
    x3 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,0,max_size)
    x4 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,0,max_size)
    x5 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,0,max_size)
    x6 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,0,max_size)
    x7 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,0,max_size)
    x8 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,0,max_size)
    x9 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,0,max_size)
    x12 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x22 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x32 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x42 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x52 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x62 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x72 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x82 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x92 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size,max_size)

    return x1, x2, x3, x4, x5, x6, x7, x8, x9, x12, x22, x32, x42, x52, x62, x72, x82, x92

def split_tesnsor_384(xp,max_size = 128):
    x1 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,0,max_size)
    x2 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,0,max_size)
    x3 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,0,max_size)
    x4 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,0,max_size)
    x5 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,0,max_size)
    x6 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,0,max_size)
    x7 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,0,max_size)
    x8 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,0,max_size)
    x9 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,0,max_size)
    x12 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x22 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x32 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x42 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x52 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x62 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x72 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x82 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x92 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x13 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x23 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x33 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x43 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x53 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x63 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x73 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x83 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x93 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size*2,max_size)

    return x1, x2, x3, x4, x5, x6, x7, x8, x9, x12, x22, x32, x42, x52, x62, x72, x82, x92, x13, x23, x33, x43, x53, x63, x73, x83, x93

def split_tesnsor_512(xp,max_size = 128):
    x1 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,0,max_size)
    x2 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,0,max_size)
    x3 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,0,max_size)
    x4 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,0,max_size)
    x5 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,0,max_size)
    x6 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,0,max_size)
    x7 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,0,max_size)
    x8 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,0,max_size)
    x9 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,0,max_size)
    x12 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x22 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x32 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x42 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x52 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x62 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x72 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x82 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x92 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x13 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x23 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x33 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x43 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x53 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x63 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x73 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x83 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x93 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x14 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x24 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x34 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x44 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x54 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x64 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x74 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x84 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x94 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)

    return x1, x2, x3, x4, x5, x6, x7, x8, x9, x12, x22, x32, x42, x52, x62, x72, x82, x92, x13, x23, x33, x43, x53, x63, x73, x83, x93, x14, x24, x34, x44, x54, x64, x74, x84, x94



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,do_bntan=True):
        super(BasicBlock, self).__init__()



        ###########################################     CONV1       ##########################################
        max_size = 128
        groups = math.ceil(inplanes / max_size)

        #dynamic convolutions with partial sums
        self.conv1 = []
        self.conv2 = []
        self.conv3 = []
        self.conv4 = []
        self.conv5 = []
        self.conv6 = []
        self.conv7 = []
        self.conv8 = []
        self.conv9 = []

        self.conv1_2 = []
        self.conv2_2 = []
        self.conv3_2 = []
        self.conv4_2 = []
        self.conv5_2 = []
        self.conv6_2 = []
        self.conv7_2 = []
        self.conv8_2 = []
        self.conv9_2 = []

        #padding outside the conv
        self.padding1 = nn.ZeroPad2d(1)
        input_dem = inplanes
        inplanes = min(input_dem,max_size)

        self.conv1 = Binaryconv1x1(inplanes, planes, stride)
        self.conv2 = Binaryconv1x1(inplanes, planes, stride)
        self.conv3 = Binaryconv1x1(inplanes, planes, stride)
        self.conv4 = Binaryconv1x1(inplanes, planes, stride)
        self.conv5 = Binaryconv1x1(inplanes, planes, stride)
        self.conv6 = Binaryconv1x1(inplanes, planes, stride)
        self.conv7 = Binaryconv1x1(inplanes, planes, stride)
        self.conv8 = Binaryconv1x1(inplanes, planes, stride)
        self.conv9 = Binaryconv1x1(inplanes, planes, stride)
        #'''
        if (input_dem > 128):     #Input channels = 256
            self.conv12 = Binaryconv1x1(inplanes, planes, stride)
            self.conv22 = Binaryconv1x1(inplanes, planes, stride)
            self.conv32 = Binaryconv1x1(inplanes, planes, stride)
            self.conv42 = Binaryconv1x1(inplanes, planes, stride)
            self.conv52 = Binaryconv1x1(inplanes, planes, stride)
            self.conv62 = Binaryconv1x1(inplanes, planes, stride)
            self.conv72 = Binaryconv1x1(inplanes, planes, stride)
            self.conv82 = Binaryconv1x1(inplanes, planes, stride)
            self.conv92 = Binaryconv1x1(inplanes, planes, stride)

        if (input_dem > 256):       #Input channels = 384
            self.conv13 = Binaryconv1x1(inplanes, planes, stride)
            self.conv23 = Binaryconv1x1(inplanes, planes, stride)
            self.conv33 = Binaryconv1x1(inplanes, planes, stride)
            self.conv43 = Binaryconv1x1(inplanes, planes, stride)
            self.conv53 = Binaryconv1x1(inplanes, planes, stride)
            self.conv63 = Binaryconv1x1(inplanes, planes, stride)
            self.conv73 = Binaryconv1x1(inplanes, planes, stride)
            self.conv83 = Binaryconv1x1(inplanes, planes, stride)
            self.conv93 = Binaryconv1x1(inplanes, planes, stride)

        if (input_dem > 384):       #Input channels = 512
            self.conv14 = Binaryconv1x1(inplanes, planes, stride)
            self.conv24 = Binaryconv1x1(inplanes, planes, stride)
            self.conv34 = Binaryconv1x1(inplanes, planes, stride)
            self.conv44 = Binaryconv1x1(inplanes, planes, stride)
            self.conv54 = Binaryconv1x1(inplanes, planes, stride)
            self.conv64 = Binaryconv1x1(inplanes, planes, stride)
            self.conv74 = Binaryconv1x1(inplanes, planes, stride)
            self.conv84 = Binaryconv1x1(inplanes, planes, stride)
            self.conv94 = Binaryconv1x1(inplanes, planes, stride)
        #'''

        ###########################################     END     ##########################################

        self.bn1 = nn.BatchNorm2d(planes)
        self.tanh1 = nn.Hardtanh(inplace=True)



        ###########################################     CONV2       ##########################################

        # max_size = 128
        inplanes = min(max_size,planes)
        groups = math.ceil(planes / max_size)
        self.padding2 = nn.ZeroPad2d(1)
        self.conv1_2 = Binaryconv1x1(inplanes, planes)
        self.conv2_2 = Binaryconv1x1(inplanes, planes)
        self.conv3_2 = Binaryconv1x1(inplanes, planes)
        self.conv4_2 = Binaryconv1x1(inplanes, planes)
        self.conv5_2 = Binaryconv1x1(inplanes, planes)
        self.conv6_2 = Binaryconv1x1(inplanes, planes)
        self.conv7_2 = Binaryconv1x1(inplanes, planes)
        self.conv8_2 = Binaryconv1x1(inplanes, planes)
        self.conv9_2 = Binaryconv1x1(inplanes, planes)
        #'''
        if(planes>128):
            self.conv12_2 = Binaryconv1x1(inplanes, planes)
            self.conv22_2 = Binaryconv1x1(inplanes, planes)
            self.conv32_2 = Binaryconv1x1(inplanes, planes)
            self.conv42_2 = Binaryconv1x1(inplanes, planes)
            self.conv52_2 = Binaryconv1x1(inplanes, planes)
            self.conv62_2 = Binaryconv1x1(inplanes, planes)
            self.conv72_2 = Binaryconv1x1(inplanes, planes)
            self.conv82_2 = Binaryconv1x1(inplanes, planes)
            self.conv92_2 = Binaryconv1x1(inplanes, planes)

        if (planes > 256):
            self.conv13_2 = Binaryconv1x1(inplanes, planes)
            self.conv23_2 = Binaryconv1x1(inplanes, planes)
            self.conv33_2 = Binaryconv1x1(inplanes, planes)
            self.conv43_2 = Binaryconv1x1(inplanes, planes)
            self.conv53_2 = Binaryconv1x1(inplanes, planes)
            self.conv63_2 = Binaryconv1x1(inplanes, planes)
            self.conv73_2 = Binaryconv1x1(inplanes, planes)
            self.conv83_2 = Binaryconv1x1(inplanes, planes)
            self.conv93_2 = Binaryconv1x1(inplanes, planes)

        if (planes > 384):
            self.conv14_2 = Binaryconv1x1(inplanes, planes)
            self.conv24_2 = Binaryconv1x1(inplanes, planes)
            self.conv34_2 = Binaryconv1x1(inplanes, planes)
            self.conv44_2 = Binaryconv1x1(inplanes, planes)
            self.conv54_2 = Binaryconv1x1(inplanes, planes)
            self.conv64_2 = Binaryconv1x1(inplanes, planes)
            self.conv74_2 = Binaryconv1x1(inplanes, planes)
            self.conv84_2 = Binaryconv1x1(inplanes, planes)
            self.conv94_2 = Binaryconv1x1(inplanes, planes)
        #'''

        self.tanh2 = nn.Hardtanh(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.do_bntan=do_bntan;
        self.stride = stride

    def forward(self, x):
        #global partial_sums

        partial_sums = []
        inplanes = x.shape[1]
        max_size = 128
        groups = math.ceil(inplanes / max_size)
        #print(groups)

        residual = x.clone()
        xp = x
        xp = self.padding1(xp)

        #splitting x
        if(xp.shape[1]<=128):
            x1,x2,x3,x4,x5,x6,x7,x8,x9 = split_tensor_128(xp)
        elif(xp.shape[1]==256):
            x1,x2,x3,x4,x5,x6,x7,x8,x9,x12,x22,x32,x42,x52,x62,x72,x82,x92= split_tesnsor_256(xp)
        elif (xp.shape[1] == 384):
            x1,x2,x3,x4,x5,x6,x7,x8,x9,x12,x22,x32,x42,x52,x62,x72,x82,x92,x13,x23,x33,x43,x53,x63,x73,x83,x93 = split_tesnsor_384(xp)
        elif (xp.shape[1] == 512):
            x1,x2,x3,x4,x5,x6,x7,x8,x9,x12,x22,x32,x42,x52,x62,x72,x82,x92,x13,x23,x33,x43,x53,x63,x73,x83,x93,x14,x24,x34,x44,x54,x64,x74,x84,x94 = split_tesnsor_512(xp)

        else:
            print(xp.shape)
            print("============ILLEGAL INPUT======================")



        out = []

        #append already quantized tensors
        out.append(self.conv1(x1))
        #print('Quantized to: {}'.format(out))
        out.append(self.conv2(x2))
        out.append(self.conv3(x3))
        out.append(self.conv4(x4))
        out.append(self.conv5(x5))
        out.append(self.conv6(x6))
        out.append(self.conv7(x7))
        out.append(self.conv8(x8))
        out.append(self.conv9(x9))

        if (xp.shape[1]>128):

            out.append(self.conv12(x12))
            out.append(self.conv22(x22))
            out.append(self.conv32(x32))
            out.append(self.conv42(x42))
            out.append(self.conv52(x52))
            out.append(self.conv62(x62))
            out.append(self.conv72(x72))
            out.append(self.conv82(x82))
            out.append(self.conv92(x92))

        if (xp.shape[1] > 256):
            out.append(self.conv13(x13))
            out.append(self.conv23(x23))
            out.append(self.conv33(x33))
            out.append(self.conv43(x43))
            out.append(self.conv53(x53))
            out.append(self.conv63(x63))
            out.append(self.conv73(x73))
            out.append(self.conv83(x83))
            out.append(self.conv93(x93))

        if (xp.shape[1] > 384):
            out.append(self.conv14(x14))
            out.append(self.conv24(x24))
            out.append(self.conv34(x34))
            out.append(self.conv44(x44))
            out.append(self.conv54(x54))
            out.append(self.conv64(x64))
            out.append(self.conv74(x74))
            out.append(self.conv84(x84))
            out.append(self.conv94(x94))

        '''
        if(inplanes>128):
            out.append(self.conv12(x1))
            out.append(self.conv22(x2))
            out.append(self.conv32(x3))
            out.append(self.conv42(x4))
            out.append(self.conv52(x5))
            out.append(self.conv62(x6))
            out.append(self.conv72(x7))
            out.append(self.conv82(x8))
            out.append(self.conv92(x9))
            out.append(self.conv13(x1))
            out.append(self.conv23(x2))
            out.append(self.conv33(x3))
            out.append(self.conv43(x4))
            out.append(self.conv53(x5))
            out.append(self.conv63(x6))
            out.append(self.conv73(x7))
            out.append(self.conv83(x8))
            out.append(self.conv93(x9))
            if(inplanes>2)
        '''

        '''
        for i in range(groups):
            if (i == (groups - 1)):
                out.append(self.conv1[i](x1.narrow(1, (max_size * i), inplanes)))
                out.append(self.conv2[i](x2.narrow(1, (max_size * i), inplanes)))
                out.append(self.conv3[i](x3.narrow(1, (max_size * i), inplanes)))
                out.append(self.conv4[i](x4.narrow(1, (max_size * i), inplanes)))
                out.append(self.conv5[i](x5.narrow(1, (max_size * i), inplanes)))
                out.append(self.conv6[i](x6.narrow(1, (max_size * i), inplanes)))
                out.append(self.conv7[i](x7.narrow(1, (max_size * i), inplanes)))
                out.append(self.conv8[i](x8.narrow(1, (max_size * i), inplanes)))
                out.append(self.conv9[i](x9.narrow(1, (max_size * i), inplanes)))
            else:
                out.append(self.conv1[i](x1.narrow(1, (max_size * i), max_size)))
                out.append(self.conv2[i](x2.narrow(1, (max_size * i), max_size)))
                out.append(self.conv3[i](x3.narrow(1, (max_size * i), max_size)))
                out.append(self.conv4[i](x4.narrow(1, (max_size * i), max_size)))
                out.append(self.conv5[i](x5.narrow(1, (max_size * i), max_size)))
                out.append(self.conv6[i](x6.narrow(1, (max_size * i), max_size)))
                out.append(self.conv7[i](x7.narrow(1, (max_size * i), max_size)))
                out.append(self.conv8[i](x8.narrow(1, (max_size * i), max_size)))
                out.append(self.conv9[i](x9.narrow(1, (max_size * i), max_size)))
        '''
        output = torch.zeros(out[0].shape).cuda()
        i = 0
        for out_tensor in out:
            if binarize and not quantize:
                out_tensor = torch.clamp(out_tensor, -thresh, thresh)
                out_tensor = scale * Binarize(out_tensor)

            output = output + out_tensor

            if graph:
                #print(graph)
                #temp.append(out_tensor.detach().cpu().numpy().flatten())
                #print(out_tensor.detach().cpu().numpy().flatten().type())
                #print(partial_sums)
                #print(out_tensor.detach().cpu().numpy().flatten())
                partial_sums = partial_sums + out_tensor.detach().cpu().numpy().flatten().tolist()






        output = self.bn1(output)
        xn = self.tanh1(output)





        xn = self.padding2(xn)
        inplanes = x.shape[1]
        groups = math.ceil(inplanes / max_size)

        if (xn.shape[1] <= 128):
            x1, x2, x3, x4, x5, x6, x7, x8, x9 = split_tensor_128(xn)
        elif (xn.shape[1] == 256):
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x12, x22, x32, x42, x52, x62, x72, x82, x92 = split_tesnsor_256(xn)
        elif (xn.shape[1] == 384):
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x12, x22, x32, x42, x52, x62, x72, x82, x92, x13, x23, x33, x43, x53, x63, x73, x83, x93 = split_tesnsor_384(
                xn)
        elif (xn.shape[1] == 512):
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x12, x22, x32, x42, x52, x62, x72, x82, x92, x13, x23, x33, x43, x53, x63, x73, x83, x93, x14, x24, x34, x44, x54, x64, x74, x84, x94 = split_tesnsor_512(
                xn)

        else:
            print(xn.shape)
            print("============ILLEGAL INPUT======================")



        out = []

        out.append(self.conv1_2(x1))
        out.append(self.conv2_2(x2))
        out.append(self.conv3_2(x3))
        out.append(self.conv4_2(x4))
        out.append(self.conv5_2(x5))
        out.append(self.conv6_2(x6))
        out.append(self.conv7_2(x7))
        out.append(self.conv8_2(x8))
        out.append(self.conv9_2(x9))

        if(xn.shape[1]>128):
            out.append(self.conv12_2(x12))
            out.append(self.conv22_2(x22))
            out.append(self.conv32_2(x32))
            out.append(self.conv42_2(x42))
            out.append(self.conv52_2(x52))
            out.append(self.conv62_2(x62))
            out.append(self.conv72_2(x72))
            out.append(self.conv82_2(x82))
            out.append(self.conv92_2(x92))

        if (xn.shape[1] > 256):
            out.append(self.conv13_2(x13))
            out.append(self.conv23_2(x23))
            out.append(self.conv33_2(x33))
            out.append(self.conv43_2(x43))
            out.append(self.conv53_2(x53))
            out.append(self.conv63_2(x63))
            out.append(self.conv73_2(x73))
            out.append(self.conv83_2(x83))
            out.append(self.conv93_2(x93))

        if (xn.shape[1] > 384):
            out.append(self.conv14_2(x14))
            out.append(self.conv24_2(x24))
            out.append(self.conv34_2(x34))
            out.append(self.conv44_2(x44))
            out.append(self.conv54_2(x54))
            out.append(self.conv64_2(x64))
            out.append(self.conv74_2(x74))
            out.append(self.conv84_2(x84))
            out.append(self.conv94_2(x94))



        output = torch.zeros(out[0].shape).cuda()

        for out_tensor in out:
            if binarize and not quantize:
                out_tensor = tensor.clamp(out_tensor, -thresh, thresh)
                out_tensor = scale * Binarize(out_tensor)

            output = output + out_tensor

            if graph:
                partial_sums = partial_sums + out_tensor.detach().cpu().numpy().flatten().tolist()



        if self.downsample is not None:
            if residual.data.max()>1:
                import pdb; pdb.set_trace()
            #print("TRUEEEEEEEEEEEEEEEEEEEEE")
            residual = self.downsample(residual)


        #print(str(output.shape) + " ==>> " + str(residual.shape))
        output += residual
        #if self.do_bntan:
        output = self.bn2(output)
        output = self.tanh2(output)

        #print(partial_sums)


        #temp = sum(temp,[])


########################## GRAPH #######################################

        if graph:
            plot_dist(partial_sums)
            #temp = partial_sums
            #print(partial_sums)
            #plot_importance(partial_sums)



        return output


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = BinarizeConv2d(inplanes, planes, kernel_size=1, bias=False, nbits=num_bit, quantize=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BinarizeConv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, nbits=num_bit, quantize=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = BinarizeConv2d(planes, planes * 4, kernel_size=1, bias=False, nbits=num_bit, quantize=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.tanh = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        import pdb; pdb.set_trace()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.tanh(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        #if self.do_bntan:
        out = self.bn2(out)
        out = self.tanh2(out)

        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, stride=1,do_bntan=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                BinarizeConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False, nbits=num_bit, quantize=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes))
        layers.append(block(self.inplanes, planes,do_bntan=do_bntan))
        return nn.Sequential(*layers)

    def forward(self, x):
        #p_s1,p_s2,p_s3,ps_4 = []
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.bn1(x)
        x = self.tanh1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)
        x = self.tanh2(x)
        x = self.fc(x)
        x = self.bn3(x)
        x = self.logsoftmax(x)

        return x


class ResNet_imagenet(ResNet):

    def __init__(self, num_classes=1000,
                 block=Bottleneck, layers=[3, 4, 23, 3]):
        super(ResNet_imagenet, self).__init__()
        self.inplanes = 64
        self.conv1 = BinarizeConv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False, nbits=num_bit, quantize=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.tanh1 = nn.Hardtanh(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.bn2 = nn.BatchNorm1d(512*block.expansion)
        self.tanh2 = nn.Hardtanh(inplace=True)

        self.fc = BinarizeLinear(512 * block.expansion, num_classes)
        self.logsoftmax = nn.LogSoftmax()
        self.bn3 = nn.BatchNorm1d(1000)
        init_model(self)
        self.regime = {
            0: {'optimizer': 'SGD', 'lr': 1e-1,
                'weight_decay': 1e-4, 'momentum': 0.9},
            30: {'lr': 1e-2},
            60: {'lr': 1e-3, 'weight_decay': 0},
            90: {'lr': 1e-4}
        }


class ResNet_cifar10(ResNet):

    def __init__(self, num_classes=10,
                 block=BasicBlock, depth=18):
        super(ResNet_cifar10, self).__init__()
        self.inflate = 4
        self.inplanes = 16*self.inflate
        n = int((depth - 2) / 6)
        self.conv1 = BinarizeConv2d(3, 16*self.inflate, kernel_size=3, stride=1, padding=1,
                               bias=False, nbits=num_bit, quantize=False)
        self.maxpool = lambda x: x
        self.bn1 = nn.BatchNorm2d(16*self.inflate)
        self.tanh1 = nn.Hardtanh(inplace=True)
        self.tanh2 = nn.Hardtanh(inplace=True)
        self.layer1 = self._make_layer(block, 16*5, n)
        self.layer2 = self._make_layer(block, 32*self.inflate, n, stride=2)
        self.layer3 = self._make_layer(block, 96*self.inflate, n, stride=2,do_bntan=False)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        self.bn2 = nn.BatchNorm1d(96*self.inflate)
        self.bn3 = nn.BatchNorm1d(10)
        self.logsoftmax = nn.LogSoftmax()
        self.fc = BinarizeLinear(96*self.inflate, num_classes)

        init_model(self)
        #self.regime = {
        #    0: {'optimizer': 'SGD', 'lr': 1e-1,
        #        'weight_decay': 1e-4, 'momentum': 0.9},
        #    81: {'lr': 1e-4},
        #    122: {'lr': 1e-5, 'weight_decay': 0},
        #    164: {'lr': 1e-6}
        #}
        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            101: {'lr': 1e-3},
            142: {'lr': 5e-4},
            184: {'lr': 1e-4},
            220: {'lr': 1e-5}
        }


def resnet_binary(**kwargs):
    num_classes, depth, dataset = map(
        kwargs.get, ['num_classes', 'depth', 'dataset'])


    if dataset == 'cifar10':
        num_classes = num_classes or 10
        depth = depth or 18
        return ResNet_cifar10(num_classes=num_classes,
                              block=BasicBlock, depth=depth)
    else:                                       #if dataset == 'imagenet':
        num_classes = num_classes or 1000
        depth = depth or 18
        if depth == 18:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=BasicBlock, layers=[2, 2, 2, 2])
        if depth == 34:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=BasicBlock, layers=[3, 4, 6, 3])
        if depth == 50:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 4, 6, 3])
        if depth == 101:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 4, 23, 3])
        if depth == 152:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 8, 36, 3])

