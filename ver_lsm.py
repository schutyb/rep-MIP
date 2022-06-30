import PhasorLibrary as phlib
import tifffile
import matplotlib.pyplot as plt


file = str('/home/bruno/Documentos/Proyectos/TESIS/TESIS/base de datos lsm/2022/MELANOMAS/scaneo lineal/HE-Melanomas/19507_SP_Tile_5x5.lsm')
m = 5
n = 5
im = tifffile.imread(file)
dc, g, s, _, _ = phlib.phasor_tile(im, 1024, 1024)
dc = phlib.concatenate(dc, n, m)
g = phlib.concatenate(g, n, m)
s = phlib.concatenate(s, n, m)
phlib.interactive(dc, g, s, 0.2)

plt.figure(1)
plt.imshow(dc, cmap='gray')
plt.axis('off')
plt.show()
