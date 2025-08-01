import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter

def normalise(a: np.array, use_abs: bool=False):
    if use_abs:
        temp = (a - np.mean(a))
        return  temp / np.abs(np.max(temp))
    else:
        return (a - np.mean(a)) / np.sqrt(np.var(a))


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5*fs
    low = lowcut / nyq 
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a 

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

if __name__ == "__main__":
    
    import pandas as pd
    dat = pd.read_csv("chb01_300_430_all_channels.csv")
    
    names = ['FP1-F7', 'F7-T7', 'T7-P7', 'FP2-F8', 'F8-T8', 'T8-P8']

    mat = dat[names].to_numpy()
    mat = np.transpose(mat)
    d, N = mat.shape
    freq = 256
    
    # seizure starts at 327 seconds, data here starts at 300 seconds => starts at 27 seconds in
    seizure_ind = freq * 27
    
    every = 3
    
    # seizure starts at 27 seconds, we analyse 15 seconds of data from 2 seconds before seizure onset
    start = 25 * freq
    end = start + 15 * freq
    
    fig, axs = plt.subplots(d, 1, sharex=True)
    vals = []
    for ind, name in enumerate(names):
        y = mat[ind, :]
        y = butter_bandpass_filter(y, 1, 30, freq) # digital filter between 1-30Hz to remove noise and isolate key rythmic frequencies
        y = np.diff(y[start:end][::every]) # downsample by factor of every and apply first order difference to improve stationarity for ar processes
        y = normalise(y) # normalise to have mean 0 and variance 1.
        
        vals.append(y)
        axs[ind].plot(np.arange(len(y)), y)
        axs[ind].vlines([int((seizure_ind-start)/every)], -10, 10, color='red')
        axs[ind].set_title(names[ind])
    plt.show()
    
    with open("eeg_dat.csv", 'w') as file:
        writer = csv.writer(file, delimiter=',')
        for val in vals:
            writer.writerow(val)
        writer.writerow(vals[0])
        