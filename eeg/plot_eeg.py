import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

if __name__ == "__main__":
    
    ind = 3 # choose which chain to use
    data = np.loadtxt(f"results/adj_{ind}.csv", delimiter=',')
    d, _ = data.shape
    data = np.ma.masked_array(data, mask=[1 if i == j else 0 for i in range(d) for j in range(d)])
    names = ['FP1-F7', 'F7-T7', 'T7-P7', 'FP2-F8', 'F8-T8', 'T8-P8']
    ticks = np.arange(d) + 0.5
    
    plt.rcParams["figure.figsize"] = [9, 8]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()
    im = ax.pcolormesh(data, cmap=cm.gray_r, edgecolors='white', linewidths=1,
                    antialiased=True)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names, rotation=90)
    ax.set_yticklabels(names)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    fig.colorbar(im)
        
    fig.savefig("images/eeg_adj.pdf", bbox_inches='tight')
    ax.patch.set(hatch='xx', edgecolor='black')

    plt.show()