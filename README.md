# Image Analysis
Using Python scripts for creating bone mask, 3D Adaptive thresholding, subchondral bone plate segmentation, and 3D analysis. 

## Required packages
* mimics  
* trimatic
* numpy
* json
* globe
* math
## Creating a bone mask
The bone masking procedure can be called in the terminal using following input:
python BoneMask.py <link to input folder> <threshold>
The user gives as first input the link to the folder of the bmp files for which they want to create bone masks and then, as second input, a global threshold that needs to be performed in able to segement the bone masks.
## 3D Adaptive thresholding 
The most accurate adaptive thresholding method for trabecular bone is using the mean value of the maximum and minimum intensity level in a predefined kernel. While there are many packages such as OpenCV (with their function adaptiveThreshold) and skimage (with their function threshold_adaptive) which have implemented functions for adaptive thresholding, none of them have the option to calculate the mean of the min and max intensity levels, and none of them are designed for 3D images. Therefore an own implementation was made.
## Separating subchondral bone plate from underneath subchondral trabecular bone 
  


## Citing
|||
|-----------------------|-----------------|
|**Author / Master student** | Emma Van Riet| 
|**Author / PhD Candidate** | Fahimeh Azari|
|**P.I.**| Harry van Lenthe |
|**Department** | Mechanical Engineering, Biomechanics Section|
|**Institution** | KU Leuven |
  
The future publication cover the technical parts of this project: **link to be added in the near future**

