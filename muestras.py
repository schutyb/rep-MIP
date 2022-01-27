"""
    con este codigo agrego el numero de la muestra a la lista
    tambien hago dos listas una de casos hechos y otra con casos no hechos
"""
import pandas as pd
import numpy as np


#  Leo el csv con la lista de los numeros de informes
df = pd.read_csv("/home/bruno/Documentos/TESIS/datos-pacientes/num_informes.csv")

#  opciones de manipulacion


#  agregar el numero de una nueva muestra
num_nuevo = int(input("Ingrese el numero de la muestra ya adquirida "))
i = np.where(df.nohechos == num_nuevo)
ind = i[0][0]
