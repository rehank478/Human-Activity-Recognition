import re
import math
from csv import writer
import numpy as np
import matplotlib.pyplot as plt
import cmath
from scipy.interpolate import interp1d
import scipy.signal as signal
import pandas as pd
import pywt
import sys
import statistics as stat
import seaborn as sn
# 
#
# wden(x, tptr, sorh, scal, n, wname) does wavelet denoising. 
#
# x, input signal to be denoised
# tptr, threshold selection rule. See thselect.
# sorh, threshold type. See wthresh
# scal = 'one', for no threshold rescaling
#      = 'sln', for rescaling using a single estimation of level noise based 
#               on the first detail coefficients
#      = 'mln', for rescaling done using level dependent estimation
# wname, wavelet name
#
def wden(x, tptr, sorh, scal, n, wname):
    
    # epsilon stands for a very small number
    eps = 2.220446049250313e-16
    
    # decompose the input signal. Symetric padding is given as a default.
    coeffs = pywt.wavedec(x, wname,'sym', n)
    
    # threshold rescaling coefficients
    if scal == 'one':
        s = 1
    elif scal == 'sln':
        s = wnoisest(coeffs)
    elif scal == 'mln':
        s = wnoisest(coeffs, level = n)
    else: 
        raise ValueError('Invalid value for scale, scal = %s' %(scal))
    
    # wavelet coefficients thresholding
    coeffsd = [coeffs[0]]
    for i in range(0, n):
        if tptr == 'sqtwolog' or tptr == 'minimaxi':
            th = thselect(x, tptr)
        else:                	
            if np.size(s) == 1:
                if s < math.sqrt(eps) * max(coeffs[1+i]): th = 0
                else: th = thselect(coeffs[1+i]/s, tptr)
            else:
                if s[i] < math.sqrt(eps) * max(coeffs[1+i]): th = 0
                else: th = thselect(coeffs[1+i]/s[i], tptr)
        
        ### DEBUG
#        print "threshold before rescaling:", th
        ###
        
        # rescaling
        if np.size(s) == 1: th = th *s
        else: th = th *s[i]
        
        #### DEBUG
#        print "threshold:", th
        ####
        
        coeffsd.append(np.array(wthresh(coeffs[1+i], sorh, th)))
        
    # wavelet reconstruction 
    xdtemp = pywt.waverec(coeffsd, wname, 'sym')
    
    # get rid of the extended part for wavelet decomposition
    extlen = int(abs(len(x)-len(xdtemp))/2)
    LENX_EXTLEN = int(len(x)+extlen)
    xd = xdtemp[extlen:LENX_EXTLEN]
    
    return xd
    
    
# ################################

# thselect(x, tptr) returns threshold x-adapted value using selection rule defined by string tptr.

# tptr = 'rigrsure', adaptive threshold selection using principle of Stein's Unbiased Risk Estimate.
#        'heursure', heuristic variant of the first option.
#        'sqtwolog', threshold is sqrt(2*log(length(X))).
#        'minimaxi', minimax thresholding. 
        
def thselect(x, tptr):
    x = np.array(x) # in case that x is not an array, convert it into an array
    l = len(x)
    
    if tptr == 'rigrsure':
        sx2 = [sx*sx for sx in absolute(x)]
        sx2.sort()
        cumsumsx2 = cumsum(sx2)        
        risks = []
        for i in xrange(0, l):
            risks.append((l-2*(i+1)+(cumsumsx2[i]+(l-1-i)*sx2[i]))/l)
        mini = argmin(risks)
        th = math.sqrt(sx2[mini])
    if tptr == 'heursure':
        hth = math.sqrt(2*math.log(l))
        
        # get the norm of x
        normsqr = np.dot(x, x)
        eta = 1.0*(normsqr-l)/l
        crit = (np.math.log(l,2)**1.5)/math.sqrt(l)
        
        ### DEBUG
#        print "crit:", crit
#        print "eta:", eta
#        print "hth:", hth
        ###
        
        if eta < crit: th = hth
        else: 
            sx2 = [sx*sx for sx in np.absolute(x)]
            sx2.sort()
            cumsumsx2 = np.cumsum(sx2)        
            risks = []
            for i in range(0, l):
                risks.append((l-2*(i+1)+(cumsumsx2[i]+(l-1-i)*sx2[i]))/l)
            mini = np.argmin(risks)
            
            
            ### DEBUG
#            print "risk:", risks[mini]
#            print "best:", mini
#            print "risks[222]:", risks[222]
            ###
            
            rth = math.sqrt(sx2[mini])
            th = min(hth, rth)     
    elif tptr == 'sqtwolog':
        th = sqrt(2*log(l))
    elif tptr == 'minimaxi':
        if l <32: th = 0
        else: th = 0.3936 + 0.1829*log(l, 2)
    else:
        raise ValueError('Invalid value for threshold selection rule, tptr = %s' %(tptr))
    
    return th
    
#################################
#
# wthresh(x, sorh, t) returns the soft (sorh = 'soft') or hard (sorh = 'hard') 
# thresholding of x, the given input vector. t is the threshold.
# sorh = 'hard', hard trehsholding
# sorh = 'soft', soft thresholding
#    
    
def wthresh(x, sorh, t):
    
    if sorh == 'hard':
        y = [e*(abs(e) >= t) for e in x]
    elif sorh == 'soft':
        y = [((e<0)*-1.0 + (e>0))*((abs(e)-t)*(abs(e) >= t)) for e in x]
    else:
        raise ValueError('Invalid value for thresholding type, sorh = %s' %(sorh))
    
    return y

#################################
#
# wnoisest(coeffs, level = None) estimates the variance(s) of the given detail(s)
#
# coeffs = [CAn, CDn, CDn-1, ..., CD1], multi-level wavelet coefficients
# level, decomposition level. None is the default.
#

def wnoisest(coeffs, level= None):
    
    l = len(coeffs) - 1
    
    if level == None:
        sig = [abs(s) for s in coeffs[-1]]
        stdc = median(sig)/0.6745
    else:
        stdc = []
        for i in xrange(0, l):
            sig = [abs(s) for s in coeffs[1+i]]
            stdc.append(median(sig)/0.6745)
    
    return stdc

#################################
#
# median(data) returns the median of data
#
# data, a list of numbers.
#       
 
def median(data):
        
    temp = data[:]
    temp.sort()
    dataLen = len(data)
    DL = int(dataLen/2)
#    print(dataLen)
    if dataLen % 2 == 0: # even number of data points
        med = (temp[DL-1] + temp[DL])/2
    else:
        med = temp[DL]
        
    return med   
    
 


if __name__ == "__main__":
    """
    This script file demonstrates how to transform raw CSI out from the ESP32 into CSI-amplitude and CSI-phase.
    """

    FILE_NAME = "./sit0.csv"

    f = open(FILE_NAME)
    next(f)
    mac1 = ""
    mac2 = ""
    mydict1 = dict()
    mydict2 = dict()
    mn = 1000
    for j, l in enumerate(f.readlines()):
        # Parse string to create integer list

        t1 = float(l.split(',')[23])
        mac  = l.split(',')[2]
        if(mac1 == ""):
            print(mac)
            mac1 = mac
        if (mac1 != "" and mac1 != mac and mac2 == ""):
            print(mac)
            mac2 = mac
        mn = min(t1, mn)
        csi_string = re.findall(r"\[(.*)\]", l)[0]
        raw = [x for x in l.split(',')]
        csi_raw = [int(x) for x in csi_string.split(" ") if x != '']
        if(mac == mac1 and mac2!=""):
            mydict1[t1] = csi_raw
        if(mac == mac2):
            mydict2[t1] = csi_raw
    
    amplitude = []
    time = []
    for data in sorted(mydict1):
        csi = mydict1[data]
        real_time = data
        img = []
        real = []
        for i in range(len(csi)):
            if(i % 2 == 0):
                img.append(csi[i])
            else:
                real.append(csi[i])
        amp = []
        for i in range(int(len(csi)/2)):
            amp.append(math.sqrt(img[i] ** 2 + real[i] ** 2))
        amplitude.append(amp)
        time.append(data)
        
    amplitude2 = []
    time2 = []
    for data in sorted(mydict2):
        csi = mydict2[data]
        real_time = data
        img = []
        real = []
        for i in range(len(csi)):
            if(i % 2 == 0):
                img.append(csi[i])
            else:
                real.append(csi[i])
        amp = []
        for i in range(int(len(csi)/2)):
            amp.append(math.sqrt(img[i] ** 2 + real[i] ** 2))
        amplitude2.append(amp)
        time2.append(data)
    print(len(amplitude))
    print(len(amplitude2))
    
    allin =[]
    
    for i in range(len(amplitude[0])):
        amp = []
        for j in range(len(amplitude)):
            x = amplitude[j][i]
            if(x == 0):
                x = 0.1
            amp.append(20*cmath.log10(x))
        f = interp1d(time, amp)
        x_new = np.linspace(np.min(time), np.max(time), int(len(time)/3))
        y_new = f(x_new)
        XD = wden(y_new,'heursure','soft','sln',5,'sym6')
        plt.plot(x_new, XD, linewidth = 0.4)
        allin.append(y_new)
    # plt.xlabel('Time(seconds)')
    # plt.ylabel('CSI Amplitude (db)')
    # plt.title(FILE_NAME)
    # plt.show()
    
    
    for i in range(len(amplitude2[0])):
        amp = []
        for j in range(len(amplitude2)):
            x = amplitude2[j][i]
            if(x == 0):
                x = 0.1
            amp.append(20*cmath.log10(x))
        f = interp1d(time2, amp)
        x_new = np.linspace(np.min(time2), np.max(time2), int(len(time)/3))
        y_new = f(x_new)
        XD = wden(y_new,'heursure','soft','sln',5,'sym6')

        allin.append(y_new)
        plt.plot(x_new, XD, linewidth = 0.4)
    plt.xlabel('Time(seconds)')
    plt.ylabel('CSI Amplitude (db)')
    plt.title(FILE_NAME)
    plt.show()
    
    print(len(allin))