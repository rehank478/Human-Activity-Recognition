import re
from math import sqrt, atan2
from csv import writer
import numpy as np
import matplotlib.pyplot as plt
import cmath
from scipy.interpolate import interp1d


if __name__ == "__main__":
    """
    This script file demonstrates how to transform raw CSI out from the ESP32 into CSI-amplitude and CSI-phase.
    """

    FILE_NAME = "./data_j.csv"

    f = open(FILE_NAME)
    next(f)
    mydict = dict()
    mn = 1000
    for j, l in enumerate(f.readlines()):
        # Parse string to create integer list

        t1 = float(l.split(',')[23])
        mn = min(t1, mn)
        csi_string = re.findall(r"\[(.*)\]", l)[0]
        raw = [x for x in l.split(',')]
        csi_raw = [int(x) for x in csi_string.split(" ") if x != '']
        mydict[t1] = csi_raw
    amplitude = []
    time = []
    for data in sorted(mydict):
        csi = mydict[data]
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
            amp.append(sqrt(img[i] ** 2 + real[i] ** 2))
        amplitude.append(amp)
        time.append(data-mn)
    
    for i in range(len(amplitude[0])):
        amp = []
        for j in range(len(amplitude)):
            x = amplitude[j][i]
            if(x == 0):
                x = 0.1
            amp.append(20*cmath.log10(x))
        f = interp1d(time, amp)
        x_new = np.linspace(np.min(time), np.max(time), len(time))
        y_new = f(x_new)
        plt.plot(x_new, y_new, linewidth = 0.4)
    plt.xlabel('Time(seconds)')
    plt.ylabel('CSI Amplitude (db)')
    plt.title(FILE_NAME)
    plt.show()
    
    