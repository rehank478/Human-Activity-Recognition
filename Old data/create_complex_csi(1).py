import re
import math as mt
from csv import writer
from numpy import *
import matplotlib.pyplot as plt
import scipy.signal as signal
import pandas as pd
import pywt
import sys
from pywt import wavedec

if __name__ == "__main__":
    """
    This script file demonstrates how to transform raw CSI out from the ESP32 into CSI-amplitude and CSI-phase.
    """
	
    FILE_NAME = "./sit.csv"

    f = open(FILE_NAME)
    next(f)
    amplitude = []
    

    for j, l in enumerate(f.readlines()):
        # Parse string to create integer list
    
        arr = []
        img = []
        real = []
        raw = []
        amp = []
        
        csi_string = re.findall(r"\[(.*)\]", l)[0]
        raw = [x for x in l.split(',')]
        csi_raw = [int(x) for x in csi_string.split(" ") if x != '']

        # Create list of imaginary and real numbers from CSI
        for i in range(len(csi_raw)):
        	if i % 2 == 0:
        		img.append(csi_raw[i])
        	else:
        		real.append(csi_raw[i])
        for i in range(int(len(csi_raw) / 2)):
            temp = sqrt(img[i] ** 2 + real[i] ** 2)
            if temp<=0:
                amp.append(temp)
            else:
                amp.append(20*(mt.log10(temp)))
        amplitude.append(amp)
    # for low band pass filtering
    '''
    N  = 3   # Filter order
    Wn = 0.1 # Cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')
    
    for i in range(int(len(amplitude[0]))):
        amp = []
        num = []
        for k in range(int(len(amplitude))):
            amp.append(amplitude[k][i])
            num.append(k)
        smooth_data = signal.filtfilt(B,A,amp)
        plt.plot(num,smooth_data)
    plt.xlabel('Packet No.')
    plt.ylabel('Amplitude')
    plt.show()
    '''
    # for moving average filtering
    """
    for i in range(int(len(amplitude[0]))):
        amp = []
        num = []
        for k in range(int(len(amplitude))):
            amp.append(amplitude[k][i])
            num.append(k)
        smooth_data = pd.Series(amp).rolling(window=7).mean().plot()
        #plt.plot(num,smooth_data)
    plt.xlabel('Packet No.')
    plt.ylabel('Amplitude')
    plt.show()
    """
    
    for i in range(int(len(amplitude[0]))):
        amp = []
        num = []
        for k in range(int(len(amplitude))):
            amp.append(amplitude[k][i])
            num.append(k)
        plt.plot(num,amp)
    plt.xlabel('Packet No.')
    plt.ylabel('Amplitude')
    plt.show()
	
    '''
    for i in range(int(len(amplitude[0]))):
        amp = []
        num = []
        for k in range(int(len(amplitude))):
            amp.append(amplitude[k][i])
            num.append(k)
        coeffs = wavedec(amp, 'db1', level=2)
        plt.plot(num,coeffs)
    plt.xlabel('Packet No.')
    plt.ylabel('Amplitude')
    plt.show()
    '''

        	
        	
        	
        	
        	
        	
        	
        	
        	
        	
        	
        	
