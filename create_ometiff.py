import PhasorLibrary as phlib
import tifffile
from skimage.filters import median, gaussian
import matplotlib.pyplot as plt
import numpy as np

file = str('/home/bruno/Documentos/Proyectos/TESIS/TESIS/base de datos lsm/comp_HE/HE-Melanomas/16952_SP_Tile_4x3.lsm')
im = tifffile.imread(file)
dc, g, s, _, _ = phlib.phasor_tile(im, 1024, 1024)

m = 4
n = 3
dc = phlib.concatenate(dc, n, m)
g = phlib.concatenate(g, n, m)
s = phlib.concatenate(s, n, m)

dc = gaussian(dc, sigma=1.2)
cont = 0
while cont < 2:
    g = median(g)
    s = median(s)
    cont = cont + 1

# umbralización
dc = np.where(dc > 1, dc, np.zeros(dc.shape))
g = np.where(dc > 1, g, np.zeros(g.shape))
s = np.where(dc > 1, s, np.zeros(s.shape))
ph = np.round(np.angle(g + s * 1j, deg=True))
md = np.sqrt(g ** 2 + s ** 2)

# guardo los datos en un ome.tiff
filename = '/home/bruno/Documentos/Proyectos/TESIS/TESIS/base de datos lsm/comp_HE/HE-Melanomas/ometiff/16952.ome.tiff'
data = phlib.generate_file(filename, [dc, g, s, md, ph])

'''fig, ax = plt.subplots(1, 2, figsize=(20, 12))
ax[0].imshow(ph, cmap='gray', interpolation='none')
ax[1].imshow(dc, cmap='gray', interpolation='none')
plt.show()'''

# filename = '/home/bruno/Documentos/Proyectos/TESIS/MIP/data/prueba/prueba.ome.tiff'
# im = tifffile.imread(filename)
