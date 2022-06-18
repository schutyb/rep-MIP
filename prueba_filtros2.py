import numpy as np
import PhasorLibrary as phlib
import tifffile
from skimage.filters import gaussian, median
import matplotlib.pyplot as plt


def comparacion(g1, s1, dc1, g2, s2, dc2):

    edc = abs(dc1 - dc2)
    eg = abs(g1 - g2)
    es = abs(s1 - s2)

    pdc = np.mean(edc ** 2)
    pg = np.mean(eg ** 2)
    ps = np.mean(es ** 2)

    p = [pdc, pg, ps]
    return p


f = str('/home/bruno/Documentos/Proyectos/TESIS/TESIS/estudio del ruido/exp_2_vs_16/16avg/1_16avg.lsm')
im = tifffile.imread(f)
dc_optimal, g_optimal, s_optimal, _, _ = phlib.phasor(im)

f1 = str('/home/bruno/Documentos/Proyectos/TESIS/TESIS/estudio del ruido/exp_2_vs_16/2avg/1_2avg.lsm')
im1 = tifffile.imread(f1)

dc, g, s, _, _ = phlib.phasor(im1)
inter = str('nearest')  # interpolation for figure plot

# gaussian filters
sigma = np.arange(0.25, 5, 0.25)
l = len(sigma) + 1
pdc = np.zeros(l)
pg = np.zeros(l)
ps = np.zeros(l)

p = comparacion(g_optimal, s_optimal, dc_optimal, g, s, dc)
pdc[0] = p[0]
pg[0] = p[1]
ps[0] = p[2]
ind = 1

for i in sigma:
    dc2 = gaussian(dc, sigma=i)
    g2 = gaussian(g, sigma=i)
    s2 = gaussian(s, sigma=i)
    p = comparacion(g_optimal, s_optimal, dc_optimal, g2, s2, dc2)
    pdc[ind] = p[0]
    pg[ind] = p[1]
    ps[ind] = p[2]
    ind = ind + 1

sigma = np.arange(0, 5, 0.25)

plt.figure(1)
plt.plot(sigma, pdc, 'rx-')
plt.title('P(sigma) Gaussian')

plt.figure(2)
plt.plot(sigma, pg, 'gx-')
plt.plot(sigma, ps, 'bx-')
plt.title('P(sigma) Gaussian')

optimal_sg = sigma[np.where(pdc == min(pdc))][0]
optimal_filtered = gaussian(dc, sigma=optimal_sg)

fig, ax = plt.subplots(1, 3, figsize=(20, 12))
ax[0].imshow(dc_optimal, cmap='gray', interpolation=inter)
ax[0].set_title('Original 16 avg')

ax[2].imshow(optimal_filtered, cmap='gray', interpolation=inter)
ax[2].set_title('Original 2 avg')

ax[1].imshow(optimal_filtered, cmap='gray', interpolation=inter)
ax[1].set_title('optimal filtered')
plt.show()

plt.show()

''' dc_gauss02 = gaussian(dc, sigma=0.2)
dc_gauss1 = gaussian(dc, sigma=1)
dc_gauss2 = gaussian(dc, sigma=2)
dc_gauss3 = gaussian(dc, sigma=3)
dc_gauss5 = gaussian(dc, sigma=5) '''

ploty = False
if ploty:
    fig, ax = plt.subplots(2, 3, figsize=(20, 12))
    ax[0, 0].imshow(dc, cmap='gray', interpolation=inter)
    ax[0, 0].set_title('Original')
    ax[0, 1].imshow(dc_gauss02, cmap='gray', interpolation=inter)
    ax[0, 1].set_title('sg = 0.2')
    ax[0, 2].imshow(dc_gauss1, cmap='gray', interpolation=inter)
    ax[0, 2].set_title('sg = 1')
    ax[1, 0].imshow(dc_gauss2, cmap='gray', interpolation=inter)
    ax[1, 0].set_title('sg = 2')
    ax[1, 1].imshow(dc_gauss3, cmap='gray', interpolation=inter)
    ax[1, 1].set_title('sg = 3')
    ax[1, 2].imshow(dc_gauss5, cmap='gray', interpolation=inter)
    ax[1, 2].set_title('sg = 5')
    plt.show()


# median filters
dc_median = median(dc)
dc_2median = median(dc_median)

if ploty:
    fig, ax = plt.subplots(1, 3, figsize=(20, 12))
    ax[0].imshow(dc, cmap='gray', interpolation=inter)
    ax[0].set_title('Original')
    ax[1].imshow(dc_median, cmap='gray', interpolation=inter)
    ax[1].set_title('median')
    ax[2].imshow(dc_2median, cmap='gray', interpolation=inter)
    ax[2].set_title('2 median')
    plt.show()

