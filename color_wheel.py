import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import prueba_2x2
from skimage.filters import median, gaussian

f1 = str('/home/bruno/Documentos/TESIS/caso_prueba/calculados/prueba2_nevo.ome.tif')
gsa = tifffile.imread(f1)
img_avg = gsa[2]
ph = gsa[3]
md = gsa[4]

# observacion hay 0 y 360 que aparecen de hacer calculos, que hay que corregir


#  genero una imagen de la cual se los valores de modulo y fase
#  crea una imagen de modulo entre 0 y uno y otra de fase entre 0 y 360
'''
img_avg = np.zeros([2, 2])
md = np.ones([2, 2])
ph = np.zeros([2, 2])

ph[0][0] = 0
ph[0][1] = 60
ph[1][0] = 120
ph[1][1] = 240
'''
####################################################################

calcular = False
if calcular:
    # voy a crear una matriz hsv donde h va de min a max en el phasor, s de 0 a 100 y v = 100
    n_ind = 360
    vec = np.linspace(245, 315, n_ind)
    vec_ind = np.linspace(0, 360, n_ind, dtype='int')
    hsv = np.zeros([361, 101, 3]) + 100
    for i in range(0, 361):
        hsv[i:i + 1, 0:, 0:1] = i
    for i in range(0, 101):
        hsv[0:361, i:i + 1, 1:2] = i

    # los valores de ph los dejo como estan
    # los valores de md los dejo entre 0 y 1 de a 0.01
    # ind_md = np.zeros(md.shape)
    ind_ph = np.zeros(ph.shape)
    for m in range(0, ph.shape[0]):
        for n in range(0, ph.shape[1]):
            if ph[m][n]:  # and md[m][n]:
                ind_ph[m][n] = vec_ind[np.argmin(abs(vec - ph[m][n]))]
                # ind_md[m][n] = round(md[m][n], 2) * 100
            else:
                ind_ph[m][n] = vec_ind[np.argmin(abs(vec - ph[m][n]))]
                # ind_md[m][n] = md[m][n]

    img_hsv = np.zeros([img_avg.shape[0], img_avg.shape[1], 3])

    for i in range(0, img_hsv.shape[0]):
        for j in range(0, img_hsv.shape[1]):
            if ind_ph[i][j]:  # and ind_md[i][j]:
                img_hsv[i, j, :3] = hsv[int(ind_ph[i][j]), 1]  # int(ind_md[i][j])]

    file = 'prueba2.ome.tif'
    data = [img_hsv]
    prueba_2x2.generate_file(file, data)

probar = True
if probar:
    f2 = str('/home/bruno/Documentos/TESIS/codigos/MIP/prueba2.ome.tif')
    img_hsv = tifffile.imread(f2)

    a = np.zeros(img_hsv.shape)
    a[0:, 0:, 0:1] = img_hsv[0:, 0:, 0:1] / 360
    a[0:, 0:, 1:2] = img_hsv[0:, 0:, 1:2]
    a[0:, 0:, 2:3] = img_hsv[0:, 0:, 2:3] / 100

    b = hsv_to_rgb(a)
    #  b *= 255
    #  b = b.astype(int)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(a)
    axs[0].set_title('img_colored')
    axs[1].imshow(img_avg, cmap='gray')
    axs[1].set_title('Img Avg')
    plt.show()
