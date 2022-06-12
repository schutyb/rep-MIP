import numpy as np
import PhasorLibrary as phlib
import tifffile
import matplotlib.pyplot as plt


f1 = str('/home/bruno/Documentos/Proyectos/TESIS/Experimentos/exp_bordes/img_1x1/lsm/exp_1x1_nevo_2.lsm')
im = tifffile.imread(f1)

# Phasor tile
dc, g, s, md, ph = phlib.phasor(im)

# con esto veo el phasor para tener una idea el max y min de la fase
# phasor_fig = phlib.phasor_plot(dc, g, s, 0, num_phasors=1, title=None, same_phasor=False)
# plt.show()

# coloracion de la imagen
ic = 10
aux = np.where(dc > ic, ph, np.mean(ph))
maxi = np.max(aux)
mini = np.min(aux)
aux = np.where(dc > ic, ph, np.zeros(dc.shape))

dif = int(360 / (maxi-mini))
arr = np.arange(int(mini), int(maxi) + dif, dif)
