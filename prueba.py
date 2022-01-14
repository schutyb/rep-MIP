"""
    This is a testing code, it calls a json file which contains:
        the average image, its histogram and a bins array,
        it also has the phasors g and s and their filtered images.

    The json is a dictionary so in k variable it is stored the dict keys.

    If control == 1 the code will plot the previously mentioned data.
"""

import matplotlib.pyplot as plt
import json

# n in the number of times we filter G and S
n = 1

file = 'gs_'+str(n)+'filt.json'
dat = open(file)
data = json.load(dat)
k = list()
for i in data.keys():
    k.append(i)

control = False
if control:

    plotty_intensity_hist = False
    if plotty_intensity_hist:
        plt.figure(1)
        plt.imshow(data[k[0]], cmap='gray')
        plt.title('Imagen promedio')

        plt.figure(2)
        plt.bar(data[k[2]][0: len(data[k[2]]) - 1], data[k[1]], width=1)

    if n != 0:
        plt.figure(3)
        plt.subplot(1, 2, 1)
        plt.imshow(data[k[3]], 'gray', interpolation='none')
        plt.title('Phasor G original')
        plt.subplot(1, 2, 2)
        plt.imshow(data[k[5]], 'gray', interpolation='none')
        plt.title('Phasor G (filtered'+str(n)+' )')

        plt.figure(4)
        plt.subplot(1, 2, 1)
        plt.imshow(data[k[4]], 'gray', interpolation='none')
        plt.title('Phasor S original')
        plt.subplot(1, 2, 2)
        plt.imshow(data[k[6]], 'gray', interpolation='none')
        plt.title('Phasor S (filtered'+str(n)+')')
        plt.show()

    if n == 0:
        plt.figure(4)
        plt.imshow(data[k[3]], cmap='gray')
        plt.title('fasor G')

        plt.figure(5)
        plt.imshow(data[k[4]], cmap='gray')
        plt.title('fasor S')
        plt.show()

