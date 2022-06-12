import os

ruta_carpetas = '/home/bruno/Documentos/TESIS/datos-pacientes'
nombres_carpetas = os.listdir(ruta_carpetas)


def buscar_archivos(ruta):
    archivos_texto = []
    archivos = os.listdir(ruta)
    for archivo in archivos:
        archivos_texto.append(archivo)
    return archivos_texto


nombres = buscar_archivos(ruta_carpetas)

for i in range(0, len(nombres)):
    print(str(nombres[i][0:6]))
