import numpy as np
import tifffile
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import PhasorLibrary as phlib


sigma = np.arange(0, 3, 0.1)
mat_psnr = []
arr_sg_opt = []
for i in range(3):
    route_2avg = str('/home/bruno/Documentos/TESIS/TESIS/filtrado/2avg/')
    file_2avg = route_2avg + str(i + 1) + str('_2avg_3x3.lsm')
    im_2avg = tifffile.imread(file_2avg)

    route_16avg = str('/home/bruno/Documentos/TESIS/TESIS/filtrado/16avg/')
    file_16avg = route_16avg + str(i + 1) + str('_16avg_3x3.lsm')
    im_16avg = tifffile.imread(file_16avg)

    for k in range(len(im_16avg)):
        dc_optimal = np.mean(im_16avg[k], axis=0)
        dc = np.mean(im_2avg[k], axis=0)
        arr_psnr = np.zeros(len(sigma))
        ind = 0
        for j in sigma:
            dc2 = gaussian(dc, sigma=j)
            arr_psnr[ind] = phlib.psnr(dc_optimal, dc2)  # guardo la psrn de la img k para los j sigmas
            ind = ind + 1

        # elimino todos los que tiene el max menor a 40 db ya que valores tipicos son entre 30 y 50 dB para 8 bits
        if max(arr_psnr) >= 40:
            mat_psnr.append(arr_psnr)
            arr_sg_opt.append(sigma[np.where(arr_psnr == max(arr_psnr))][0])


plt.figure(1)
for i in range(len(mat_psnr)):
    plt.plot(sigma, mat_psnr[i], 'x-')
plt.title('PSNR(sigma)')
plt.xlabel('Sigma (0 ; 4.75)')
plt.ylabel('PSNR[dB]')
plt.grid()

prueba = True
if prueba:
    # regiones interesantes img 2 indice 3, 4 y 5
    optimal_sg = np.mean(arr_sg_opt)
    print('The optimal sigma for gaussian kernel is:', optimal_sg)
    im_2avg = tifffile.imread(str('/home/bruno/Documentos/TESIS/TESIS/filtrado/2avg/2_2avg_3x3.lsm'))
    im_16avg = tifffile.imread(str('/home/bruno/Documentos/TESIS/TESIS/filtrado/16avg/2_16avg_3x3.lsm'))
    ind = 3
    optimal_filtered = gaussian(np.mean(im_2avg[ind], axis=0), sigma=optimal_sg)
    inter = str('none')

    fig, ax = plt.subplots(2, 3, figsize=(20, 12))
    ax[0, 0].imshow(np.mean(im_16avg[ind], axis=0), cmap='gray', interpolation=inter)
    ax[0, 0].set_title('Original 16 avg')
    ax[0, 1].imshow(optimal_filtered, cmap='gray', interpolation=inter)
    ax[0, 1].set_title('optimal filtered')
    ax[0, 2].imshow(np.mean(im_2avg[ind], axis=0), cmap='gray', interpolation=inter)
    ax[0, 2].set_title('Original 2 avg')
    ax[1, 0].imshow(np.mean(im_16avg[ind], axis=0)[250:600, 300:800], cmap='gray', interpolation=inter)
    ax[1, 0].set_title('Original 16 avg ROI')
    ax[1, 1].imshow(optimal_filtered[250:600, 300:800], cmap='gray', interpolation=inter)
    ax[1, 1].set_title('optimal filtered ROI')
    ax[1, 2].imshow(np.mean(im_2avg[ind], axis=0)[250:600, 300:800], cmap='gray', interpolation=inter)
    ax[1, 2].set_title('Original 2 avg ROI')
    plt.show()
