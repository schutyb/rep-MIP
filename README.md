# MIP
codigos de la tesis

funciones.py
	-Contiene las funciones que se describen a continuación 
		(1) phasor: con la fft obtiene las coordenadas del phasor G y S 
		(2) data_phasor: devuelve un dict con los datos que usa para cargar y hacer la parte 
				interactiva hasta calcular g, s y y las g y s filtradas.
		(3) ph_md: calcula modulo y fase con los g y s que se le da de entrada 
		(4) histogram_line: parte interaactrica se pincha en dos puntos del plot del phasor y 
				se obtiene el histograma sobre la linea que los une.
		(5) interactive: esta funcion es la que se llama y se le pasa los datos g y s y el histograma
				que ya estaban guardados. On el hist se cortan los pixeles por debajo de un 					determinado valor, y con g y s luego se obtiene el phasor. Ademas muestra la imagen 				promedio y la img promedio de pseudocolor. 
				
json_creator.py
	-Crea un archivo json para almacenar el g y el s calculados. 

main.py 
	-Llama a la funciones interactivas que muestran el phasor. 

metadata.py 
	-Es un codigo para concatenar los tiles y armar la imagen para luego hacer el g y s, es provisorio ya que 		hay que buscar una forma de hacerlo sin concatenar.

prueba.py
	-llama el json y hace una prueba graficando
	
read_MSW.py 
	-abre el archivo word y saca el numero de la muestra.
