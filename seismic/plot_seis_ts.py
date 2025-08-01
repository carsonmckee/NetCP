import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime, timedelta

if __name__ == "__main__":
    ind = 1
    dat = np.loadtxt("seis_dat.csv", delimiter=',')
    U = np.loadtxt(f"results/out_{ind}.csv", delimiter=',')
    
    d, n = dat.shape
    
    times = np.arange(datetime(2004, 9, 28, 22, 32, 10), datetime(2004, 9, 28, 22, 33, 50), timedelta(seconds=1/20)).astype(datetime)[1:]
    
    earthquakes_inds = [112, 616, 826,  1233]
    earthquakes_times = [times[i-1] for i in earthquakes_inds]
    
    fs = 14
    
    names = ['FROB', 'VCAB', 'CCRB', 'SMNB', 'MMNB', 'LCCB', 'SCYB', 'RMNB'][:d]
    
    inds = [0, 2, 4, 7]
    inds = range(d)
    
    for i, ind in enumerate(inds):
        name = names[ind]
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 2), height_ratios=(3,1))
        axs[0].plot(times, dat[ind, 1:], linewidth=0.5, color='black')
        axs[0].vlines(earthquakes_times, -100, 100, color='blue', linestyle='dashed')
        axs[0].set_ylabel(name, fontsize=fs, rotation=90)
        axs[0].set_ylim(-10, 10)
        axs[1].stem(times, U[ind, :], 'red', markerfmt=" ", basefmt=" ")
        stem = axs[1].stem(times, U[ind, :], 'red', markerfmt=" ", basefmt=" ")
        stem[1].set_linewidth(0.05)
        axs[1].set_ylim(0, 1)
        
        if i == (d-1):
            plt.xticks(fontsize=9)
            axs[1].set_xlabel('Time', fontsize=fs)
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
    