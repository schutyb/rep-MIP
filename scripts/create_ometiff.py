import PhasorLibrary as phlib
import tifffile
from skimage.filters import median, gaussian
import numpy as np
import os


def create_phasor_file(image=None, path=None, many=False, one=False, m=None, n=None, path2=None, filename=None):
    if one:
        dc, g, s, _, _ = phlib.phasor_tile(image, 1024, 1024)
        dc = phlib.concatenate(dc, n, m)
        g = phlib.concatenate(g, n, m)
        s = phlib.concatenate(s, n, m)
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
            dc = phlib.concatenate(dc, n, m)
            g = phlib.concatenate(g, n, m)
            s = phlib.concatenate(s, n, m)
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
