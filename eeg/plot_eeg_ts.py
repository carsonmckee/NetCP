import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime, timedelta

if __name__ == "__main__":
    ind = 1
    dat = np.loadtxt("eeg_dat.csv", delimiter=',')
    U = np.loadtxt(f"results/out_{ind}.csv", delimiter=',')
    
    d, n = dat.shape
    print(dat.shape)
    print(U.shape)
    
    fs = 14
    
    names = ['FP1-F7', 'F7-T7', 'T7-P7', 'FP2-F8', 'F8-T8', 'T8-P8']
    
    inds = [0, 1, 3, 4]
    inds = range(d)
    times = np.arange(n-1)
    
    start = 325
    end = 340
    
    dx = (end-start) / n
    
    times = []
    for i in range(n):
        times.append(start + i*dx)
    print(times)
    print(len(times))
    print(n)
    for i, ind in enumerate(inds):
        name = names[ind]
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 2), height_ratios=(3,1))
        axs[0].plot(times[1:], dat[ind, 1:], linewidth=0.5, color='black')
        axs[0].vlines([327], -100, 100, color='blue', linestyle='dashed')
        axs[0].set_ylabel(name, fontsize=fs, rotation=90)
        axs[0].set_ylim(np.min(dat)-1, np.max(dat)+1)
        axs[1].stem(times[1:], U[ind, :], 'red', markerfmt=" ", basefmt=" ")
        stem = axs[1].stem(times[1:], U[ind, :], 'red', markerfmt=" ", basefmt=" ")
        stem[1].set_linewidth(0.05)
        axs[1].set_ylim(0, 1)
        
        if i == (d-1):
            plt.xticks(fontsize=9)
            axs[1].set_xlabel('Time (Seconds)', fontsize=fs)
        else:
            plt.tick_params(
                        axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=False) # labels along the bottom edge are off
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        fig.autofmt_xdate()
        fig.savefig(f"images/{name}.pdf", bbox_inches='tight', format='pdf')
        plt.show()
    