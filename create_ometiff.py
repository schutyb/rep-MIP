import PhasorLibrary as phlib
import tifffile
from skimage.filters import median, gaussian
import numpy as np
import os


im = tifffile.imread('/home/bruno/Documentos/Proyectos/TESIS/TESIS/Data/lsm/Nevos/15237_SP_Tile_6x5.lsm')
m = 6
n = 5

dc, g, s, _, _ = phlib.phasor_tile(im, 1024, 1024)
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
filename = '/home/bruno/Documentos/Proyectos/TESIS/TESIS/Data/ometiff/Nevos/prueba.ome.tiff'
data = phlib.generate_file(filename, [dc, g, s, md, ph])


if False:
    path = '/home/bruno/Documentos/Proyectos/TESIS/TESIS/Data/lsm/Nevos'
    files_names = os.listdir(path)
    for k in range(len(files_names)):
        im = tifffile.imread(path + files_names[k])
        dc, g, s, _, _ = phlib.phasor_tile(im, 1024, 1024)
        muestra = files_names[k][0:5]
        m = int(files_names[k][14:16])
        n = int(files_names[k][17:19])
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
        filename = '/home/bruno/Documentos/Proyectos/TESIS/TESIS/Data/ometiff/Melanomas/' + muestra + '.ome.tiff'
        data = phlib.generate_file(filename, [dc, g, s, md, ph])

