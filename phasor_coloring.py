import numpy as np
import PhasorLibrary
import tifffile
import matplotlib.pyplot as plt
import cv2


f1 = str('/home/bruno/Documentos/TESIS/caso_prueba/calculados/prueba2_nevo.ome.tif')
gsa = tifffile.imread(f1)
g = gsa[0]
s = gsa[1]
img_avg = gsa[2]

plotty = False
if plotty:
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(g)
    axs[0].set_title('G')
    axs[1].imshow(s)
    axs[1].set_title('S')
    axs[2].imshow(img_avg)
    axs[2].set_title('Img Avg')
    plt.show()

'''
#  voy a hacer el phasor
Ro = 0.1
ic = 5
x_c = []
y_c = []

"""store the coordinate to plot in the phasor"""
for i in range(0, len(g)):
    for j in range(0, len(g)):
        if img_avg[i][j] > ic:
            x_c.append(g[i][j])
            y_c.append(s[i][j])
            
'''

g2 = np.concatenate(g)
s2 = np.concatenate(s)
X1 = np.zeros([2, len(g2)])
X1[0:, 0:] = g2, s2
X2 = X1.T
X = X2[~np.isnan(X2).any(axis=1)]
x_aux = X[0:, 0:1]
y_aux = X[0:, 1:2]
x = np.concatenate(x_aux)
y = np.concatenate(y_aux)
Ro = 0.1
ic = 5

phasor = True
if phasor:
    while True:
        PhasorLibrary.interactive2(img_avg, x, y, Ro, g, s, ic)

