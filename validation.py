import lfdfiles
import json
import matplotlib.pyplot as plt
import numpy as np

with lfdfiles.SimfcsR64('imagen2.R64') as f:
    data = f.asarray()

# store the phases and modulation for the 1st
cd = data[0]
ph1 = data[1]
md1 = data[2]

data_exp = open('datos_prueba/pm_1filt_gs.json')
data_exp = json.load(data_exp)
k = list()
for i in data_exp.keys():
    k.append(i)

cd_exp = np.asarray(data_exp[k[0]])
ph1_exp = np.asarray(data_exp[k[1]])
md1_exp = np.asarray(data_exp[k[2]])

titulos = ['Intensidad-python', 'ph1-python', 'md1-python', 'Intensidad-SimFCS', 'ph1-SimFCS', 'md1-SimFCS']

# Corrijo la escala de color de ph1 SimFSC
data[1] = np.where(data[1] > 0, data[1], 80)

# corrijo la escala de color de la img de intensidad de python
for i in range(0, 256):
    for j in range(0, 256):
            if 900 < data_exp[k[0]][i][j] < 1500:
                data_exp[k[0]][i][j] = 90
            if data_exp[k[0]][i][j] > 1500:
                data_exp[k[0]][i][j] = 5

plotty = True
if plotty:
    for i in range(0, 3):
        plt.figure(i)
        plt.title(titulos[i])
        plt.imshow(data_exp[k[i]], cmap='gray')

    for i in range(0, 3):
        plt.figure(i + 3)
        plt.title(titulos[i + 3])
        plt.imshow(data[i], cmap='gray')

# Mapas de calor Python vs SimFCS
plot_map = True
if plot_map:
    x1 = np.concatenate(data[1])
    y1 = np.concatenate(np.asarray(data_exp[k[1]]))
    plt.figure(7)
    plt.hist2d(x1, y1, bins=512, cmap='magma')
    cb = plt.colorbar()
    cb.set_label('Number of pixels')
    plt.xlabel('SimFCS')
    plt.ylabel('Python')
    plt.title('Phase')

    x2 = np.concatenate(data[2])
    y2 = np.concatenate(np.asarray(data_exp[k[2]]))
    plt.figure(8)
    plt.hist2d(x2, y2, bins=512, cmap='magma')
    cb = plt.colorbar()
    cb.set_label('Number of pixels')
    plt.xlabel('SimFCS')
    plt.ylabel('Python')
    plt.title('Modulation')

    plt.show()


# import napari


# create a Viewer and add an image here
# viewer = napari.view_image(np.asarray(data_exp[k[1]]))
# start the event loop and show the viewer
# napari.run()

