import funciones
import json


dat = open('data.json')
data = json.load(dat)
k = list()
for i in data.keys():
    k.append(i)

img_mean = data[k[0]]
hist_img = data[k[1]]
bins = data[k[2]]
G = data[k[3]]
S = data[k[4]]
G_filt = data[k[5]]
S_filt = data[k[6]]

Ro = 0.1

"""APLICO LAS FUNCIONES"""

_, x_c, y_c = funciones.interactive(img_mean, hist_img, bins, G_filt, S_filt, Ro)
#components_histogram = funciones.histogram_line(Ro, x_c, y_c, 100, G_filt, S_filt, img_mean, Ro, print_fractions=False,
                                               # plot_histogram=True)
