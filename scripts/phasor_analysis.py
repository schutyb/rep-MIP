import numpy as np
import tifffile
import matplotlib.pyplot as plt
import PhasorLibrary as PhLib

from matplotlib import cm
import matplotlib as mpl


rgb = tifffile.imread('/home/bruno/Documentos/TESIS/TESIS/Modelado/rgb/rgb.ome.tiff')
im = tifffile.imread('/home/bruno/Documentos/TESIS/TESIS/Modelado/rgb/rgb-15719.ome.tiff')
hist_mod = tifffile.imread('/home/bruno/Documentos/TESIS/TESIS/Modelado/hist_mod.ome.tiff')
hist_ph = tifffile.imread('/home/bruno/Documentos/TESIS/TESIS/Modelado/hist_phase.ome.tiff')
ph = tifffile.imread('/home/bruno/Documentos/TESIS/TESIS/Modelado/hist_phase.ome.tiff')
mod = tifffile.imread('/home/bruno/Documentos/TESIS/TESIS/Modelado/hist_phase.ome.tiff')

per = 0.001
auxph = PhLib.segment_thresholding(hist_ph[0], hist_ph[1], per)
auxmd = PhLib.segment_thresholding(hist_mod[0], hist_mod[1], per)
phinterval = np.asarray([min(auxph), max(auxph)])
mdinterval = np.asarray([min(auxmd), max(auxmd)])

r = np.geomspace(mdinterval[0], mdinterval[1], num=int(len(auxmd)))
th = np.linspace(phinterval[0], phinterval[1], num=int(len(auxph)))
th = np.radians(th)
color = np.zeros([rgb.shape[0], rgb.shape[1]])

for i in range(rgb.shape[0]):
    for j in range(rgb.shape[1]):
        color[i][j] = (rgb[i][j][0]*6/256)*36 + (rgb[i][j][1]*6/256)*6 + (rgb[i][j][2]*6/256)

color = np.transpose(color)

