import tifffile
import cv2
import matplotlib.pyplot as plt
import numpy as np

f = str('imagen.lsm')
im = tifffile.imread(f)



imagen = np.zeros([30, 4120, 4120])


img = np.zeros([im.shape[0], im.shape[1], 824, 824])

for j in range(8, 11):

    for i in range(0, 25):
        img[i][j] = im[i][j][100:924, 100:924]
    
    i1 = cv2.hconcat([img[0][0], img[1][j]])
    i2 = cv2.hconcat([img[9][0], img[8][j]])
    i3 = cv2.hconcat([img[10][0], img[11][j]])
    i4 = cv2.hconcat([img[19][0], img[18][j]])
    i5 = cv2.hconcat([img[20][0], img[21][j]])

    for i in range(2, 5):
        i1 = cv2.hconcat([i1, img[i][j]])
        i2 = cv2.hconcat([i2, img[9-i][j]])
        i3 = cv2.hconcat([i3, img[i+10][j]])
        i4 = cv2.hconcat([i4, img[19-i][j]])
        i5 = cv2.hconcat([i5, img[i+20][j]])

    aux1 = cv2.vconcat([i1, i2])
    aux2 = cv2.vconcat([aux1, i3])
    aux3 = cv2.vconcat([aux2, i4])
    aux4 = cv2.vconcat([aux3, i5])

    imagen[j] = aux4
"""
    for i in range(0, aux4.shape[0]):
        for p in range(0, aux4.shape[1]):
            if aux4[i][p] > 256:
                aux4[i][p] = 0
"""




#plt.imshow(aux4)
#plt.show()
