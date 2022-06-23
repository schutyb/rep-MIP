import PhasorLibrary as phlib
import tifffile
import matplotlib.pyplot as plt


file = str('/home/bruno/Documentos/Proyectos/TESIS/TESIS/base de datos lsm/2021/20338_SP_Tile_6x4_pos_2a.lsm')
m = 6
n = 4
im = tifffile.imread(file)
dc, g, s, _, _ = phlib.phasor_tile(im, 1024, 1024)
dc = phlib.concatenate(dc, n, m, bidirectional=True)
# g = phlib.concatenate(g, n, m, bidirectional=True)
# s = phlib.concatenate(s, n, m, bidirectional=True)
# phlib.interactive(dc, g, s, 0.2)

plt.figure(1)
plt.imshow(dc, cmap='nipy_spectral')
plt.show()
