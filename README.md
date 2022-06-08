# Melanoma identification through phasor analysis 

### This scripts were made to compute phasor analysis with HSI of skin tissue.
This repository contains all the algorithms and scripts to compute the phasor analysis 
in hyperspectral imaging of skin tissue. In this project it was developed many tools 
to process the image stack acquired by a Zeiss 880 confocal microscopy. It includes a 
concatenating algorithms to solve tile images stitching and alignment. Histogram 
thresholding, image filtering. 


#### PhasorLibrary.py
Main library that contains the functions developed to do the image processing and 
analysis for this project. 

#### noise_simulation.py
This script simulate and study the noise at the image. Given a hyperspectral image
it is simulated the process of stitching 5% of the border where there is overlap in
the image of each tile to obtain the complete microscopy image. This process is done 
in two ways, concatenating the tile first and the computing the phasor, and vice versa.  

