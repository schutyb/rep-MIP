"""
    this python file calls data_phasor which calculate
    img mean, hist, bins, g, s, and the median filtered
    of g and s returns them in dict and stores them in
    a json file.

    INPUT:
        use a lsm image which is the input of the function
        data_phasor.
"""

import codecs
import json
import funciones

f = str('imagen.lsm')
n_harmonic = 1
filt_times = 0
filt_gs = 1

data_phasor_bool = True
if data_phasor_bool:
    dict = funciones.data_phasor(f, n_harmonic, filt_times)
    with open('data_5x5.json', 'w') as fp:
        json.dump(dict, fp, indent=4)

ph_md_bool = False
if ph_md_bool:
    dict_ph_md = funciones.ph_md(f, n_harmonic, filt_times, filt_gs, ph2_conditional=False)
    with open('datos_prueba/pm_1filt_gs.json', 'w') as fp_pm:
        json.dump(dict_ph_md, fp_pm, indent=4)
