import obspy 
import matplotlib.pyplot as plt
import numpy as np
import csv
from obspy.signal.filter import bandpass
from statsmodels.graphics.tsaplots import plot_acf

from scipy.signal import butter, lfilter, freqz, decimate, filtfilt

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

def bandpass_filter(data, fs=20.0, lowcut=1.0, highcut=16.0, order=4):
    """
    Apply a Butterworth bandpass filter between 1 and 16 Hz
    for a signal sampled at 20 Hz.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    if high >= 1.0:
        raise ValueError("Highcut frequency must be less than Nyquist (fs/2).")

    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

if __name__ == '__main__':

    st = obspy.read("seis_dat_raw.mseed")
    
    print(st)
    
    names = ['CCRB', 'EADB', 'FROB', 'GHIB', 'JCNB', 'JCSB', 'LCCB', 'MMNB', 'RNMB', 'SCYB', 'SMNB', 'VARB', 'VCAB']
    use_stations = ['FROB', 'VCAB', 'CCRB', 'SMNB', 'MMNB', 'LCCB', 'SCYB']
    inds = [names.index(station) for station in use_stations]
    vals = []
    every = 1
    since = 200
    upto = 2000
    fig, axs = plt.subplots(len(inds), 1, sharex=True)
    times = st[0].times('utcdatetime')[since:][:upto] 
    
    print([times[0], times[-1]])
    for k, i in enumerate(inds):
        dat = bandpass(st[i].data, 2, 16, 10)[since:] # previsouly working
        dat = dat[:upto]
        
        dat = (dat - np.mean(dat)) / np.sqrt(np.var(dat))
        vals.append(dat)
        axs[k].plot_date(times, dat, color='black')
        print(dat)
    
    # with open("seis_dat.csv", 'w') as file:
    #     writer = csv.writer(file)
    #     for i in range(len(inds)):
    #         writer.writerow(vals[i])
    
    plt.show()
    