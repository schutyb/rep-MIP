import PhasorLibrary as phlib
import tifffile
from skimage.filters import median, gaussian
import numpy as np
import os


def create_phasor_file(image=None, one=False, fname=None, many=False, path=None, path2=None, filename=None,
                       bidirectional=False):
    """
    :param image: nd-array. HSI image to compute the phasor
    :param fname: string, image file name
    :param bidirectional: set true if the tile was acquire bidirectionally
    :param path: string. Contains the path to teh folder where the image is store
    :param many: set true to use all the HSI images in the path
    :param one: set true to compute only one phasor
    :param path2: string, path to store the generated file
    :param filename: the name of the new file
    :return: None
    """
    if one:
        dc, g, s, _, _ = phlib.phasor_tile(image, 1024, 1024)
        m = int(fname[14:16])
        n = int(fname[17:19])
        dc = phlib.concatenate(dc, m, n, bidirectional=bidirectional)
        g = phlib.concatenate(g, m, n, bidirectional=bidirectional)
        s = phlib.concatenate(s, m, n, bidirectional=bidirectional)
        dc = gaussian(dc, sigma=1.2)
        cont = 0
        while cont < 2:
            g = median(g)
            s = median(s)
            cont = cont + 1
        # thresholding
        dc = np.where(dc > 1, dc, np.zeros(dc.shape))
        g = np.where(dc > 1, g, np.zeros(g.shape))
        s = np.where(dc > 1, s, np.zeros(s.shape))
        ph = np.round(np.angle(g + s * 1j, deg=True))
        md = np.sqrt(g ** 2 + s ** 2)
        # store the data as ome.tiff
        f = path2 + filename + '.ome.tiff'
        phlib.generate_file(f, [dc, g, s, md, ph])

    if many:
        files_names = os.listdir(path)
        for k in range(len(files_names)):
            im = tifffile.imread(path + files_names[k])
            dc, g, s, _, _ = phlib.phasor_tile(im, 1024, 1024)
            m = int(files_names[k][14:16])
            n = int(files_names[k][17:19])
            dc = phlib.concatenate(dc, n, m, bidirectional=bidirectional)
            g = phlib.concatenate(g, n, m, bidirectional=bidirectional)
            s = phlib.concatenate(s, n, m, bidirectional=bidirectional)
            dc = gaussian(dc, sigma=1.2)
            cont = 0
            while cont < 2:
                g = median(g)
                s = median(s)
                cont = cont + 1
            # thresholding
            dc = np.where(dc > 1, dc, np.zeros(dc.shape))
            g = np.where(dc > 1, g, np.zeros(g.shape))
            s = np.where(dc > 1, s, np.zeros(s.shape))
            ph = np.round(np.angle(g + s * 1j, deg=True))
            md = np.sqrt(g ** 2 + s ** 2)
            # store the data as ome.tiff
            f = path2 + filename + '.ome.tiff'
            phlib.generate_file(f, [dc, g, s, md, ph])
    return None


path = '/home/bruno/Documentos/Proyectos/Tesis/Datos del Modelo/Nevo Intradermico/HSI/'
fname = '15220_SP_Tile_05x04_bidir.lsm'
filename = '15220'
path2 = '/home/bruno/Documentos/Proyectos/Tesis/Datos del Modelo/Nevo Intradermico/Phasor/'
im = tifffile.imread(path + fname)
create_phasor_file(im, one=True, fname=fname, filename=filename, path2=path2, bidirectional=True)
