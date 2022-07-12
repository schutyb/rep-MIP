import numpy as np
import PhasorLibrary
import tifffile
import matplotlib.pyplot as plt


#  voy a importar las 4 imagenes y luego calcular G y S de una sola, de dos promedidas y de 4 promediadas.
f1 = str('/home/bruno/Documentos/TESIS/TESIS/estudio del ruido/exp avg/experimental_noise/test_2mean_1.lsm')
im1 = tifffile.imread(f1)
f2 = str('/home/bruno/Documentos/TESIS/TESIS/estudio del ruido/exp avg/experimental_noise/test_2mean_2.lsm')
im2 = tifffile.imread(f2)
f3 = str('/home/bruno/Documentos/TESIS/TESIS/estudio del ruido/exp avg/experimental_noise/test_2mean_3.lsm')
im3 = tifffile.imread(f3)
f4 = str('/home/bruno/Documentos/TESIS/TESIS/estudio del ruido/exp avg/experimental_noise/test_2mean_4.lsm')
im4 = tifffile.imread(f4)

#  hago la distribucion de un pixel para los 4 G y S
ity1 = np.zeros(30)
ity2 = np.zeros(30)
ity3 = np.zeros(30)
ity4 = np.zeros(30)
lam = np.linspace(400, 700, 30)

for i in range(0, 30):
    ity1[i] = im1[i][512][512]
    ity2[i] = im2[i][512][512]
    ity3[i] = im3[i][512][512]
    ity4[i] = im4[i][512][512]

plt.figure(1)
plt.plot(lam, ity1)
plt.plot(lam, ity2)
plt.plot(lam, ity3)
plt.plot(lam, ity4)
plt.xlabel(r'Wave length [nm]')
plt.ylabel(r'I$(\lambda)$')
plt.grid()
plt.show()
