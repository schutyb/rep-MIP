import numpy as np
import tifffile
import matplotlib.pyplot as plt


his_mod = tifffile.imread('/home/bruno/Documentos/Proyectos/TESIS/TESIS/Data/ometiff/hist_mod.ome.tiff')
his_ph = tifffile.imread('/home/bruno/Documentos/Proyectos/TESIS/TESIS/Data/ometiff/hist_phase.ome.tiff')

plotty = True
if plotty:
    plt.figure(1)
    plt.plot(his_mod[1], his_mod[0])
    plt.yscale('log')

    plt.figure(2)
    plt.plot(his_ph[1], his_ph[0])
    plt.yscale('log')

    plt.show()

