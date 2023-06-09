# Image Analysis
TUsing Python scripts as an open source image analysis tool which can deal with images of photon-counting CT technique and thus confirm the bone microstructure imaging.
This tool mainly looks at three image processing steps, namely the extraction of the bone mask required for the registration, 3D Adaptive thresholding, subchondral bone plate segmentation, and 3D analysis where the
tools main focus lies on trabecular bone parameters. 

## Required packages
* numpy  

## Creating a bone mask
The BoneMask script operates as follows:
* Read the bmp images using the OpenCV function "imread".
* Use the glob module to locate and consolidate all .bmp files in the input folder into a list.
* Convert the list to a numpy array for simultaneous processing of the images.
* Threshold the 3D image using the SimpleITK function Threshold.
* Separate the different bones using the ConnectedComponent image filter from SimpleITK.
* Order the labels according to the size of the objects using the RelabelComponent function.
* Segment the four desired bones (femur, tibia, fibula, and patella).
* Apply a bilateral filter to remove remaining speckles and noise.
* Implement the shrink-wrap plugin from CTAn using an active contour algorithm.
* Preprocess the image by enhancing the visibility of the contours with inverse Gaussian gradient.

The morphological geodesic active contour function then starts from a defined region of interest (ROI) and shrinks the boundaries to fit the bone boundaries. The process is accelerated by performing morphological dilation and connectivity filtering on the thresholded and despeckled image. Finally, the morphological_geodesic_active_contour function is called with the necessary parameters to segment the bone contours.

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

