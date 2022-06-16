import numpy as np
import PhasorLibrary as Ph
import tifffile
from skimage.filters import gaussian, sato, hessian, median
# import matplotlib.pyplot as plt


f1 = str('/home/bruno/Documentos/Proyectos/TESIS/TESIS/Experimentos/exp_avg/test_16_mean.lsm')
im = tifffile.imread(f1)
g, s, _, _, dc = Ph.phasor(im, harmonic=1)


def comparacion(g, s, dc, im2, prt=False, filter_name="No name"):
    g2, s2, _, _, dc2 = Ph.phasor(im2, harmonic=1)

    eg = abs(g - g2)
    es = abs(s - s2)
    edc = abs(dc - dc2)

    pg = np.mean(eg ** 2)
    ps = np.mean(es ** 2)
    pdc = np.mean(edc ** 2)

    if prt:
        print("Filter name:", filter_name)
        print('Potencia del error en g', pg)
        print('Potencia del error en s', ps)
        print('Potencia del error en dc', pdc)

    p = [pg, ps, pdc]
    err = [eg, es, edc]
    return err, p


# Comparo ambos casos para tener idea del error error inicial y luego filtrar la imagen y ver si disminuyo
f2 = str('/home/bruno/Documentos/Proyectos/TESIS/TESIS/Experimentos/exp_avg/2_avg/test_2mean_1.lsm')
im2 = tifffile.imread(f2)

err_inicial, _ = comparacion(g, s, dc, im2, prt=True, filter_name="Original")

" Aplico diferentes tipos de filtros y loc comparo con el GS"

# Gauss
im_gauss = gaussian(im2)
err_gauss, _ = comparacion(g, s, dc, im_gauss, prt=True, filter_name="Gauss")

# Sato
im_sato = sato(im2)
err_sato, _ = comparacion(g, s, dc, im_sato, prt=True, filter_name="Sato")

# Hessian
im_hessian = hessian(im2)
err_hessian, _ = comparacion(g, s, dc, im_hessian, prt=True, filter_name="Hessian")

# Median
im_median = median(im2)
err_median, _ = comparacion(g, s, dc, im_median, prt=True, filter_name="Median")

