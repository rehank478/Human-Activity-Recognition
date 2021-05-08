import re
import math as math
from csv import writer
import numpy as np
import scipy.signal as signal
import pandas as pd
import pywt
import sys
import statistics as stat
from scipy.stats import moment, kurtosis, skew

def haarFWT ( signal, level ):

    s = .5;                 

    h = [ 1,  1 ];          
    g = [ 1, -1 ];                 
    f = len ( h );           

    t = signal;             
    l = len ( t );           
    y = [0] * l;             
    pd = np.linspace(0,0,int(len(signal)))
    t = t + pd;       

    for i in range ( level ):

        y [ 0:l ] = [0] * l; 
        l2 = l // 2;        

        for j in range ( l2 ):            
            for k in range ( f ):                
                y [j]    += t [ 2*j + k ] * h [ k ] * s;
                y [j+l2] += t [ 2*j + k ] * g [ k ] * s;

        l = l2;              
        t [ 0:l ] = y [ 0:l ] ;

    return y



def wden(x, tptr, sorh, scal, n, wname):
    
    eps = 2.220446049250313e-16
    
    coeffs = pywt.wavedec(x, wname,'sym', n)
    
    if scal == 'one':
        s = 1
    elif scal == 'sln':
        s = wnoisest(coeffs)
    elif scal == 'mln':
        s = wnoisest(coeffs, level = n)
    else: 
        raise ValueError('Invalid value for scale, scal = %s' %(scal))
    
    
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
        
        
        if np.size(s) == 1: th = th *s
        else: th = th *s[i]
        
        
        
        coeffsd.append(np.array(wthresh(coeffs[1+i], sorh, th)))
        
    xdtemp = pywt.waverec(coeffsd, wname, 'sym')
   
    extlen = int(abs(len(x)-len(xdtemp))/2)
    LENX_EXTLEN = int(len(x)+extlen)
    xd = xdtemp[extlen:LENX_EXTLEN]
    
    return xd
 
def thselect(x, tptr):
    x = np.array(x) 
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
        
      
        normsqr = np.dot(x, x)
        eta = 1.0*(normsqr-l)/l
        crit = (np.math.log(l,2)**1.5)/math.sqrt(l)

        if eta < crit: th = hth
        else: 
            sx2 = [sx*sx for sx in np.absolute(x)]
            sx2.sort()
            cumsumsx2 = np.cumsum(sx2)        
            risks = []
            for i in range(0, l):
                risks.append((l-2*(i+1)+(cumsumsx2[i]+(l-1-i)*sx2[i]))/l)
            mini = np.argmin(risks)
        
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
    
def wthresh(x, sorh, t):
    
    if sorh == 'hard':
        y = [e*(abs(e) >= t) for e in x]
    elif sorh == 'soft':
        y = [((e<0)*-1.0 + (e>0))*((abs(e)-t)*(abs(e) >= t)) for e in x]
    else:
        raise ValueError('Invalid value for thresholding type, sorh = %s' %(sorh))
    return y


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

def median(data):
        
    temp = data[:]
    temp.sort()
    dataLen = len(data)
    DL = int(dataLen/2)
    if dataLen % 2 == 0: 
        med = (temp[DL-1] + temp[DL])/2
    else:
        med = temp[DL]
        
    return med   
    
 

if __name__ == "__main__":
    """
    This script file demonstrates how to transform raw CSI out from the ESP32 into CSI-amplitude and CSI-phase.
    """
	
    FILE_NAME = ".\walk4.csv"
    sr = 'walk'
    f = open(FILE_NAME)
    next(f)
    amplitude = []
    

    for j, l in enumerate(f.readlines()):
    
        arr = []
        img = []
        real = []
        raw = []
        amp = []
        
        csi_string = re.findall(r"\[(.*)\]", l)[0]
        raw = [x for x in l.split(',')]
        csi_raw = [int(x) for x in csi_string.split(" ") if x != '']

        for i in range(len(csi_raw)):
        	if i % 2 == 0:
        		img.append(csi_raw[i])
        	else:
        		real.append(csi_raw[i])
        for i in range(int(len(csi_raw) / 2)):
            if i%1==0: 
                temp = math.sqrt(img[i] ** 2 + real[i] ** 2)
                if temp<=0:
                    amp.append(0.000001)
                else:
                    amp.append(20*math.log10(temp))
        amplitude.append(amp)
    print(len(amplitude),len(amplitude[0]))
    Denoised = []
    mean = []
    stddev = []
    ind =0
    for i in range(int(len(amplitude[0]))):
        amp = []
        num = []
        for k in range(int(len(amplitude))):
            amp.append(amplitude[k][i])
            num.append(k)
        XD = wden(amp,'heursure','soft','sln',5,'sym6')
        mean.append((sum(XD)/len(XD)))
        stddev.append(stat.stdev(XD, mean[i]))
        Denoised.append(XD)
    
    print(len(Denoised),len(Denoised[0]))
    
    standardized = []

    for i in range(int(len(Denoised))): 
        t=[]
        for o in range(int(len(Denoised[0]))): 
            temp = (Denoised[i][o] - mean[i]) / stddev[i]
            t.append(temp)
        standardized.append(t)
    data = np.array(standardized)

    covMatrix = np.cov(data,bias=True)

    values, vectors = np.linalg.eig(covMatrix)

    lam_sum = sum(values)
 
    eig_pairs = [(np.abs(values[i]), vectors[:,i]) for i in range(len(values))]

    matrix_w = np.hstack((eig_pairs[3][1].reshape(64,1),
                        eig_pairs[4][1].reshape(64,1),
                        eig_pairs[5][1].reshape(64,1),
                        eig_pairs[6][1].reshape(64,1),
                        eig_pairs[7][1].reshape(64,1),
                        eig_pairs[8][1].reshape(64,1),))
    
    Y = (data.T).dot(matrix_w)
    y=(Y.real).T
    features = []
    for i in range(0, 6):
        temp = []
        mean = sum(y[i]) / len(y[i])
        temp.append(mean)
        median = stat.median(y[i])
        temp.append(median)
        stdd = stat.stdev(y[i])
        temp.append(stdd)
        q3, q1 = np.percentile(y[i], [75 ,25])
        iqr = q3 - q1
        temp.append(iqr)
        scm = moment(y[i], moment=2)
        temp.append(scm)
        tcm = moment(y[i], moment=3)
        temp.append(tcm)
        kurt=kurtosis(y[i])
        temp.append(kurt)
        sk = skew(y[i])
        temp.append(sk) 
        features.append(temp)


    freqs, psd = signal.welch(y) 
    haart = []
    for i in range(int(len(psd))):
        hwt = haarFWT(psd[i],5)
        haart.append(hwt)
    
    print(len(haart[0]))

    for i in range(0, 6):
        mean = sum(haart[i]) / len(haart[i])
        features[i].append(mean)
        mx = max(haart[i])
        features[i].append(mx)
        stdd = stat.stdev(haart[i])
        features[i].append(stdd)
        q3, q1 = np.percentile(haart[i], [75 ,25])
        iqr = q3 - q1
        features[i].append(iqr)
        kurt=kurtosis(haart[i])
        features[i].append(kurt)
        sk = skew(haart[i])
        features[i].append(sk) 
        features[i].append(sr)
        # print(features[i])
    file_name = "svm_data.csv"
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # header = ['SIG_MEAN','SIG_MEDIAN','SIG_STDD','SIG_IQR','SIG_SCM','SIG_TCM','SIG_KURT','SIG_SKEW','PSD_MEAN','PSD_MAX','PSD_STDD','PSD_IQR','PSD_KURT','PSD_SKEW','LABEL']
        # Add contents of list as last row in the csv file
        # csv_writer.writerow(header)
        for h in range(0,6):
            csv_writer.writerow(features[h])


    
    
  