import numpy as np
from matplotlib import pyplot as plt
import networkx as nx

if __name__ == "__main__":
    
    ind = 2
    A = np.loadtxt(f"results/adj_{ind}.csv", delimiter=',')
    W = data = np.loadtxt(f"results/W_{ind}.csv", delimiter=',')

    print((A > 0.5) * W)
    
    d, _ = A.shape
    
    positions = [(1484, 705), (895, 863), (588, 1456), (153, 1605), (1335, 1465), (1185, 1768), (744, 2221)]
    names = ['FROB', 'VCAB', 'CCRB', 'SMNB', 'MMNB', 'LCCB', 'SCYB']
    
    label_pos = positions
    
    quake_labels = ["1 & 3", "2", "4"]
    quake_locs = [(2526, 121), (2369, 414), (1778, 718)]
    
    G = nx.DiGraph()
    for name, position in zip(names, positions):
        G.add_node(name, pos=position)

    edge_alphas = []
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            if A[i, j] > 0.5:
                G.add_edge(names[i], names[j], weight=W[i, j])
                edge_alphas.append(1)
            
           
    
    plt.rcParams["figure.figsize"] = [10, 10]
    plt.rcParams["figure.autolayout"] = True
    im = plt.imread("images/seis_map.png")
    fig, ax = plt.subplots()
    im = ax.imshow(im, extent=[0, 2701, 0, 2405])
    x = np.array(range(300))
    plt.axis('off')
    
    
    pos = {name:loc for name, loc in zip(names, positions)}
    nx.draw_networkx_nodes(G,
                            nodelist=names,
                            alpha = 0,
                            pos=pos,
                            node_size=400,
                            # extent=extent,
                            node_color='orange')
    
    
    nx.draw_networkx_edges(G,
            nodelist=names,
            pos=pos,
            node_size=500,
            alpha = edge_alphas,
            edge_color='black',
            width=3)
    
    edge_labels = {(u, v) : round(G[u][v]['weight'], 2) for u,v in G.edges}
    
    for name, position in zip(names, label_pos):
        x, y = position
        if name != 'SMNB':
            y += 45
            x += 5
        else:
            y -= 85
            x -= 30
        ax.text(x, y, name, size=14)
    
    for i, (label, (x, y)) in enumerate(zip(quake_labels, quake_locs)):
        if i == 0:
            x -= 100
            y += 90
        else:
            x -= 20
            y += 80
        ax.text(x, y, label, size=20)
    
    fig.savefig("images/seismic.pdf", bbox_inches='tight', format='pdf')
    plt.show()
    plt.close()