import numpy as np
import tifffile
import matplotlib.pyplot as plt
import PhasorLibrary as phlib
from PhasorLibrary import phasor_plot


'''
    In this script it is made a comparison between all the averages obtain in the microscope
    none avg, twice, 4, 8 and 16 times, taking 16 times avg as the gold standard. 
    It is used the PSNR as the metric for the error evaluation. 
'''
route = str('/home/bruno/Documentos/Proyectos/TESIS/MIP/data/experimental averaging/')
avg = [1, 2, 4, 8, 16]
g = []
s = []
dc = []

for i in avg:
    file = route + str(i) + str('avg.lsm')
    im = tifffile.imread(file)
    dc_aux, g_aux, s_aux, _, _ = phlib.phasor(im)
    g.append(g_aux)
    s.append(s_aux)
    dc.append(dc_aux)

g = np.asarray(g)
s = np.asarray(s)
dc = np.asarray(dc)

arr_psrn_g = np.zeros(4)
arr_psrn_s = np.zeros(4)

for i in range(4):
    arr_psrn_g[i] = phlib.psnr(g[4], g[i])
    arr_psrn_s[i] = phlib.psnr(s[4], s[i])

print('PSNR for G:', arr_psrn_g)
print('PSNR for S:', arr_psrn_s)

xlabel = ['PSNR G ' + str(round(arr_psrn_g[0])) + 'dB' + '\n' + 'PSNR S ' + str(round(arr_psrn_s[0])) + 'dB',
          'PSNR G ' + str(round(arr_psrn_g[1])) + 'dB' + '\n' + 'PSNR S ' + str(round(arr_psrn_s[1])) + 'dB',
          'PSNR G ' + str(round(arr_psrn_g[2])) + 'dB' + '\n' + 'PSNR S ' + str(round(arr_psrn_s[2])) + 'dB',
          'PSNR G ' + str(round(arr_psrn_g[3])) + 'dB' + '\n' + 'PSNR S ' + str(round(arr_psrn_s[3])) + 'dB',
          'Gold Standard']
xlabel = np.asarray(xlabel)
ic = np.zeros(len(dc))
titles = ['No averaging', 'Twice averaging', '4 times averaging', '8 times averaging', '16 times averaging']
phasor_plot(dc, g, s, ic, title=titles, same_phasor=False, xlabel=xlabel)
plt.show()

