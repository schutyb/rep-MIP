import PhasorLibrary as phlib
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import os

#  con los files de ome.tiff voy a hacer un loop donde obtenga el hist de la fase y otro del módulo
#  cada vez que calcule un hist de la fase y otro del modulo lo voy sumando al siguiente así tengo los
#  histogramas totales para fase y modulo.


path = '/home/bruno/Documentos/Proyectos/TESIS/TESIS/Data/ometiff/'
files_names = os.listdir(path)

bins_ph = np.arange(360)
bins_md = np.linspace(0, 1, 501)
hist_ph = np.zeros([len(bins_ph) - 1])
hist_md = np.zeros([len(bins_md) - 1])

for k in range(len(files_names)):
    phase = np.concatenate(tifffile.imread(path + files_names[k])[4])
    modulo = np.concatenate(tifffile.imread(path + files_names[k])[3])
    hist_ph = hist_ph + np.histogram(phase, bins_ph)[0]
    hist_md = hist_md + np.histogram(modulo, bins_md)[0]

hist_ph[0] = 0
hist_md[0] = 0
bins_ph = bins_ph[:len(bins_ph) - 1]
bins_md = bins_md[:len(bins_md) - 1]

hist_phase = np.asarray([hist_ph, bins_ph], dtype=float)
hist_mod = np.asarray([hist_md, bins_md], dtype=float)
fname_phase = '/home/bruno/Documentos/Proyectos/TESIS/TESIS/Data/ometiff/hist_phase.ome.tiff'
fname_mod = '/home/bruno/Documentos/Proyectos/TESIS/TESIS/Data/ometiff/hist_mod.ome.tiff'
data1 = phlib.generate_file(fname_phase, hist_phase)
data2 = phlib.generate_file(fname_mod, hist_mod)

