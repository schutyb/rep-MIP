import numpy as np
import PhasorLibrary as phlib
import tifffile
import matplotlib.pyplot as plt


f1 = str('/home/bruno/Documentos/Proyectos/TESIS/TESIS/base de datos lsm/comp_HE/HE-Melanomas/ometiff/16952.ome.tiff')
im = tifffile.imread(f1)
ph = im[4]

