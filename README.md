# MIP
codigos de la tesis

PhasorLibrary.py

def phasor(image_stack):
   """
   :param image_stack: is a file with spectral mxm images to calculate the fast fourier transform from
   numpy library.
   :return: g: is mxm image with the real part of the fft.
   :return: s: is mxm imaginary with the real part of the fft.
   :return: md: numpy.ndarray  It is the modulus obtain with Euclidean Distance.
   :return: ph: is the phase between g ans s in degrees.
   """



def generate_file(filename, gsa):
   """
   :param filename: Type string characters. The name of the file to be written, with the extension ome.tiff
   :param gsa: Type n-dimensional array holding the data to be stored. It usually has the mxm images of
   g,s and the average image.
   :return file: The created file storing the data. If the filename extension was ome.tiff the file is an
   ome.tiff format.
   """

def concat_d2(im):
   """
   :param im: stack image with the images to be concatenated. It is a specific o 2x2 concatenation.
   :return: im_concat it is an image stack with the concatenated images.
   """

def ndmedian(im, filttime=0):
   """
   :param im: ndarray usually an image to be filtered.
   :param filttime: numbers of time to be filtered im.
   :return: ndarray with the filtered image.  """
