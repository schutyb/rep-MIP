import numpy as np
import PhasorLibrary as phlib
import tifffile
import matplotlib.pyplot as plt

'''
    Datos para hacer las pruebas
'''
f1 = str('/home/bruno/Documentos/TESIS/TESIS/Experimentos/exp_bordes/img_1x1/lsm/exp_1x1_nevo_2.lsm')
# f1 = str('/home/bruno/Documentos/TESIS/TESIS/IMAGENES LSM/2022/MELANOMAS/15719_SP_Tile_4x4.lsm')
im = tifffile.imread(f1)

# dc1, _, _, _, ph1 = phlib.phasor_tile(im, 1024, 1024)
# dc = phlib.concatenate(dc1, 4, 4)
# ph = phlib.concatenate(ph1, 4, 4)

# Phasor tile
dc, g, s, md, ph = phlib.phasor(im)


# con esto veo el phasor para tener una idea el max y min de la fase
# phasor_fig = phlib.phasor_plot(dc, g, s, 0, num_phasors=1, title=None, same_phasor=False)
# plt.show()


def rgb_phase_coloring(dc, ph, ic):
    aux = np.where(dc > ic, ph, np.mean(ph))
    maxi = np.max(aux)
    mini = np.min(aux)
    aux = np.where(dc > ic, ph, np.zeros(dc.shape))

    arr = np.linspace(int(mini), int(maxi), 4)

    img_new = np.copy(dc)
    cmap = plt.cm.gray
    norm = plt.Normalize(img_new.min(), img_new.max())
    rgba = cmap(norm(img_new))

    red = np.where(aux < arr[1], aux, np.zeros(dc.shape))
    red = np.where(red != 0)
    rgba[red[0], red[1], :3] = 1, 0, 0

    green = np.where(aux > arr[2], np.zeros(dc.shape), aux)
    green = np.where(aux < arr[1], np.zeros(dc.shape), green)
    green = np.where(green != 0)
    rgba[green[0], green[1], :3] = 0, 1, 0

    blue = np.where(aux > arr[2], aux, np.zeros(dc.shape))
    blue = np.where(blue != 0)
    rgba[blue[0], blue[1], :3] = 0, 0, 1

    return rgba


for ic in range(1, 7):
    rgba = rgb_phase_coloring(dc, ph, ic)

    plt.figure(ic)
    plt.imshow(rgba)

plt.show()
