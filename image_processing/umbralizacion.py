import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from skimage.filters import threshold_otsu
import os


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


path = '/home/bruno/Documentos/Proyectos/TESIS/TESIS/base de datos lsm/Escaner/PNG/'
files_names = os.listdir(path)

for k in range(len(files_names)):
    image = mpimg.imread('/home/bruno/Documentos/Proyectos/TESIS/TESIS/base de datos lsm/Escaner/PNG/' + files_names[k])
    image2 = rgb2gray(image)

    thresh = threshold_otsu(image2)
    binary = image2 > thresh
    aux = np.zeros(image.shape)

    for i in range(binary.shape[0]):
        for j in range(binary.shape[1]):
            if not(binary[i][j]):
                aux[i][j] = image[i][j]

    plt.imsave(('/home/bruno/Documentos/Proyectos/TESIS/TESIS/base de datos lsm/Escaner/umbralizados/' +
                files_names[k]), aux)


