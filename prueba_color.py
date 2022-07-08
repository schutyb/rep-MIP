import numpy as np
import tifffile
import matplotlib.pyplot as plt
import PhasorLibrary as PhLib
import colorsys


hist_mod = tifffile.imread('/home/bruno/Documentos/TESIS/TESIS/Modelado/hist_mod.ome.tiff')
hist_ph = tifffile.imread('/home/bruno/Documentos/TESIS/TESIS/Modelado/hist_phase.ome.tiff')

per = 0.001
auxph = PhLib.md_ph_thresholding(hist_ph[0], hist_ph[1], per)
auxmd = PhLib.md_ph_thresholding(hist_mod[0], hist_mod[1], per)

plotty = False
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
hsv = np.ones([len(auxph), len(auxmd), 2])
rgb = np.zeros(hsv.shape)

for i in range(len(auxph)):
    for j in range(len(auxmd)):
        hsv[i][j][0] = (auxph[i] - (min(auxph))) / (abs(max(auxph) - min(auxph)))
        hsv[i][j][1] = (auxmd[j] - (min(auxmd) - 0.05)) / abs(max(auxmd) - min(auxmd))
        # rgb[i][j][:] = colorsys.hsv_to_rgb(hsv[i][j][0], hsv[i][j][1], 1)

'''Fig = plt.figure()
plt.imshow(rgb)
plt.axis('off')
# plt.show()'''

plt.clf()
fig = plt.figure(figsize=[5, 5])
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
thetas_radians = np.arange(0,2.01*np.pi,np.pi/100.)
radii = np.arange(0,101,1)
ax.pcolormesh(thetas_radians, radii, hsv, edgecolors='face')
