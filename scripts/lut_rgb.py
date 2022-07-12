"""
CREATE THE LOOK-UP TABLE WITH RGB DATA. THIS LUT IS RELATED TO THE MODEL OF THE DATA
GIVE THE GENERAL HISTOGRAM OF THE COLLECTION OF YOUR HSI IMAGES, WE USE THE MODULATION AND PHASE HISTOGRAM
TO CUT OFF VALUES. THE HISTOGRAM CUT OFF IS DONE WITH SEGMENT_THRESHOLDING FUNCTION FROM PHASORPY LIBRARY
"""

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import PhasorLibrary as PhLib
import colorsys
import csv


hist_mod = tifffile.imread('/home/bruno/Documentos/Proyectos/TESIS/TESIS/Data/Modelo_melanomas/hist_mod.ome.tiff')
hist_ph = tifffile.imread('/home/bruno/Documentos/Proyectos/TESIS/TESIS/Data/Modelo_melanomas/hist_phase.ome.tiff')

per = 0.001
auxph = PhLib.segment_thresholding(hist_ph[0], hist_ph[1], per)
auxmd = PhLib.segment_thresholding(hist_mod[0], hist_mod[1], per)

# store the maximum and minimum of the segment that was given after the thresholding
phinterval = np.asarray([min(auxph), max(auxph)])
mdinterval = np.asarray([min(auxmd), max(auxmd)])

plotty = True
if plotty:
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].bar(hist_mod[1], hist_mod[0], width=0.1, align='edge')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('Modulation')
    ax[0].set_xlim(0, 1)
    ax[0].plot([auxmd[0], auxmd[0]], [0, max(hist_mod[0])], 'r')
    ax[0].plot([auxmd[len(auxmd) - 1], auxmd[len(auxmd) - 1]], [0, max(hist_mod[0])], 'r')

    ax[1].bar(hist_ph[1], hist_ph[0])
    ax[1].set_xlabel('Phase [degrees]')
    ax[1].set_xlim(20, 200)
    ax[1].set_yscale('log')
    ax[1].plot([auxph[0], auxph[0]], [0, max(hist_ph[0])], 'r')
    ax[1].plot([auxph[len(auxph) - 1], auxph[len(auxph) - 1]], [0, max(hist_ph[0])], 'r')

    print('Modulation interval', '[', str(round(min(auxmd), 4)), ',', str(round(auxmd[len(auxmd) - 1], 4)), ']')
    print('Phase interval', '[', str(round(min(auxph), 4)), ',', str(round(auxph[len(auxph) - 1], 4)), ']')

    #  con las dimensiones y los rangos de mod y phase creo la LUT
    hsv = np.ones([len(auxph), len(auxmd), 3])
    rgb = np.zeros(hsv.shape)
    for i in range(len(auxph)):
        for j in range(len(auxmd)):
            hsv[i][j][0] = 0.95 * (auxph[i] - min(auxph)) / abs(max(auxph) - min(auxph))
            hsv[i][j][1] = (auxmd[j] - min(auxmd)) / abs(max(auxmd) - min(auxmd))
            rgb[i][j][:] = colorsys.hsv_to_rgb(hsv[i][j][0], hsv[i][j][1], 1)

    # rgb lut plot
    xlabels = np.zeros(10)
    for i in range(len(xlabels)):
        xlabels[i] = round(auxmd[i * int(len(auxmd) / (len(xlabels)))], 4)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.imshow(rgb, interpolation='none', extent=[0, 180, phinterval[0], phinterval[1]])
    ax2.set_aspect(2)
    ax2.set_xlabel('Modulation')
    ax2.set_ylabel('Phase')
    ax2.set_xticklabels(xlabels)
    plt.show()

    # store the rgb image into ome.tiff and the parameters of the model into csv
    store = False
    if store:
        # store the rgb image
        filename = '/home/bruno/Documentos/Proyectos/TESIS/TESIS/Data/Modelo_melanomas/rgb.ome.tiff'
        PhLib.generate_file(filename, rgb)
        # store the values of the model
        header = ['phase min', 'phase max', 'modulation min', 'modulation max']
        data = [round(min(auxph)), round(auxph[len(auxph) - 1]), round(min(auxmd), 4), round(auxmd[len(auxmd) - 1], 4)]
        with open('/home/bruno/Documentos/Proyectos/TESIS/TESIS/Data/Modelo_melanomas/model_parameters.csv', 'w',
                  encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(data)
