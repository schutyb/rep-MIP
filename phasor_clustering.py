import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage.filters import median
import PhasorLibrary
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
from sklearn.datasets import make_blobs
from itertools import cycle
from matplotlib import colors


f1 = str('/home/bruno/Documentos/TESIS/codigos/MIP/nevo_1x1.ome.tif')
gsa = tifffile.imread(f1)
g_aux = gsa[0]
s_aux = gsa[1]
img_avg = gsa[2]

filt = False
if filt:
    i = 0
    while i < 2:
        g_aux = median(g_aux)
        s_aux = median(s_aux)
        i = i + 1

#  g2 = np.concatenate(g_aux)
#  s2 = np.concatenate(s_aux)

x1 = []
y1 = []
ic = 5

"""store the coordinate to plot in the phasor"""
for i in range(0, img_avg.shape[0]):
    for j in range(0, img_avg.shape[1]):
        if img_avg[i][j] > ic:
            x1.append(g_aux[i][j])
            y1.append(s_aux[i][j])
        else:
            x1.append(2)
            y1.append(2)

X1 = np.zeros([2, len(x1)])
X1[0:, 0:] = x1, y1
X2 = X1.T
X = X2[~np.isnan(X2).any(axis=1)]

x1 = np.linspace(start=-1, stop=1, num=500)
y_positive = lambda x1: np.sqrt(1 - x1 ** 2)
y_negative = lambda x1: -np.sqrt(1 - x1 ** 2)
x2 = np.linspace(start=-0.5, stop=0.5, num=500)
y_positive2 = lambda x2: np.sqrt(0.5 ** 2 - x2 ** 2)
y_negative2 = lambda x2: -np.sqrt(0.5 ** 2 - x2 ** 2)
x3 = np.linspace(start=-1, stop=1, num=30)
x4 = np.linspace(start=-0.7, stop=0.7, num=30)


clustering = True
if clustering:

    y = KMeans(n_clusters=3).fit_predict(X)

    plot_cluster = False
    if plot_cluster:
        plt.figure(1)
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.title("Phasor clustering")
        plt.xlabel('G')
        plt.ylabel('S')
        plt.plot(x1, list(map(y_positive, x1)), color='darkgoldenrod')
        plt.plot(x1, list(map(y_negative, x1)), color='darkgoldenrod')
        plt.plot(x2, list(map(y_positive2, x2)), color='darkgoldenrod')
        plt.plot(x2, list(map(y_negative2, x2)), color='darkgoldenrod')
        plt.scatter(x3, [0] * len(x3), marker='_', color='darkgoldenrod')
        plt.scatter([0] * len(x3), x3, marker='|', color='darkgoldenrod')
        plt.scatter(x4, x4, marker='_', color='darkgoldenrod')
        plt.scatter(x4, -x4, marker='_', color='darkgoldenrod')
        #plt.show()


#  grafico el phasor con g y s
phasor = False
if phasor:
    x_aux = X[0:, 0:1]
    y_aux = X[0:, 1:2]
    x = np.concatenate(x_aux)
    y = np.concatenate(y_aux)

    plt.figure(2)
    plt.title('Phasor')
    plt.hist2d(x, y, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
    plt.xlabel('G')
    plt.ylabel('S')

    plt.plot(x1, list(map(y_positive, x1)), color='darkgoldenrod')
    plt.plot(x1, list(map(y_negative, x1)), color='darkgoldenrod')
    plt.plot(x2, list(map(y_positive2, x2)), color='darkgoldenrod')
    plt.plot(x2, list(map(y_negative2, x2)), color='darkgoldenrod')
    plt.scatter(x3, [0] * len(x3), marker='_', color='darkgoldenrod')
    plt.scatter([0] * len(x3), x3, marker='|', color='darkgoldenrod')
    plt.scatter(x4, x4, marker='_', color='darkgoldenrod')
    plt.scatter(x4, -x4, marker='_', color='darkgoldenrod')


coloring = True
if coloring:
    #  armo una imagen a partir de y con los indices de cada grupo

    dim = int(np.sqrt(len(y)))
    y_img = np.zeros([dim, dim])
    for i in range(0, dim):
        y_img[i:i+1, 0:][0] = y[dim*i:dim*i+dim]

plt.figure(3)
plt.imshow(y_img, cmap='nipy_spectral')
plt.title("Spectral coloring image")
plt.show()

plt.figure(4)
plt.imshow(y_img, cmap='Blues')
plt.title("Blue")

plt.figure(5)
plt.imshow(y_img, cmap='Reds')
plt.title("Red")

plt.figure(6)
plt.imshow(y_img, cmap='Greens')
plt.title("Green")
plt.show()
