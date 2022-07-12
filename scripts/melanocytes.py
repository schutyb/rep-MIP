"""
Here we made an study of melanocytes, given three regions of a melanoma where we know there are melanocytes
we get the HSI images and do:
1 - Do phasor analysis; with g and s plot the phasor and obtain the colored image.
2 - There is a channel image to compared with the colored image.
3 - Plot also the segmented H&E image
4 - Plot the spectral intensity graph of some ROI's
"""

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import PhasorLibrary as PhLib
import colorsys
import csv


# 1 - Phasor analysis: phasor plot and colored image
path1 = '/home/bruno/Escritorio/MIP-Data/Melanocytes/ometiff/'
im_borde = tifffile.imread(path1 + '16952_SP_mel_bo.ome.tiff')  # ROI with border
im_r1 = tifffile.imread(path1 + '16952_SP_mel_R1.ome.tiff')  # RIO One
im_r2 = tifffile.imread(path1 + '16952_SP_mel_R2.ome.tiff')  # RIO two


# 2 - Melanocytes channel images
path1 = '/home/bruno/Escritorio/MIP-Data/Melanocytes/lsm/channel/'
im_borde_ch = tifffile.imread(path1 + '16952_channel_mel_borde.lsm')[1]
im_r1_ch = tifffile.imread(path1 + '16952_channel_mel_R1.lsm')[1]
im_r2_ch = tifffile.imread(path1 + '16952_channel_mel_R2.lsm')[1]

plot_channel = True
if plot_channel:
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    ax[0].imshow(im_borde_ch, cmap='gray')
    ax[0].axis('off')
    ax[1].imshow(im_r1_ch, cmap='gray')
    ax[1].axis('off')
    ax[2].imshow(im_r2_ch, cmap='gray')
    ax[2].axis('off')
    plt.show()

# 3 - Plot the H&E ROI

# Overlap the channel image and the colored hsi

# 4 - Plot the spectral intensity graph of some ROI's
