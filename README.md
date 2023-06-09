# Image Analysis
TUsing Python scripts as an open source image analysis tool which can deal with images of photon-counting CT technique and thus confirm the bone microstructure imaging.
This tool mainly looks at three image processing steps, namely the extraction of the bone mask required for the registration, 3D Adaptive thresholding, subchondral bone plate segmentation, and 3D analysis where the
tools main focus lies on trabecular bone parameters. The aim of these Python implementations was to provide alternatives and comparisons to the calculations performed in CTAn, allowing for a comprehensive analysis of trabecular bone morphology.

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
`python BoneMask.py <link to input folder> <threshold>`
The user gives as first input the link to the folder of the bmp files for which they want to create bone masks and then, as second input, a global threshold that needs to be performed in able to segement the bone masks.
## 3D Adaptive thresholding 
3D adaptive thresholding provides more accurate trabecular bone morphology. However, existing packages like OpenCV and skimage do not have the required functionality for calculating the mean of the min and max intensity levels or processing 3D images. Therefore, a custom implementation was developed. The process involves loading CT scan slices, pre-thresholding, creating cubes around each voxel to calculate threshold values, and handling edge cases.  
There are two scripts here for implementing segmentation. The first script, "Separation_STB.py," follows specific steps using SimpleITK, Numpy, and OpenCV libraries. It can be executed from the terminal with inputs such as the folder link, bone type (femur, tibia, or both), radius, kernel sizes, and threshold values: `python Separation_STB.py <link to input folder> <bone> <radius> <kernel sizes> <threshold>`. The output will be stored in subfolders named "bone_STB" and "bone_wholemask" within the input folder.
                                         

The second script, "Adaptivethresh3D.py," utilizes SimpleITK, Numpy, OpenCV, and Skimage libraries. It performs 3D adaptive thresholding and can be called from the terminal with inputs including the folder link, radius, and threshold: `python Adaptivethresh3D.py <link to input folder> <radius> <threshold>`. The output, thresholded images, will be saved in a subfolder named "3D_adaptive_thresh" within the input folder.
                                           
## Separating subchondral bone plate from underneath subchondral trabecular bone 
It involves 2D adaptive thresholding, separation of subchondral trabecular bone and subchondral bone plate, and the creation of periosteal and endosteal masks. Various image processing techniques such as dilation, subtraction, and morphological closing are used. The resulting masks are saved as BMP files for further analysis.
  
## 3D Analysis   
  
This section provides a detailed explanation of Python-implemented methods for calculating morphometric parameters such as bone volume fraction (BV/TV), trabecular thickness (Tb.Th), trabecular separation (Tb.Sp), and trabecular number (Tb.N). The Tb.Th and Tb.Sp parameters are grouped together as they share the same calculation method, with the only difference being the use of a binary inverted image for Tb.Sp.

BV/TV:
The BV/TV calculation was implemented in Python using the PyimageJ module, which allows running BoneJ's volume fraction function in a headless environment. The implementation is based on voxel counting, comparing the number of foreground voxels (representing bone) to the total number of voxels in the image.

Trabecular thickness and trabecular separation:
Various implementations were compared with CTAn for calculating Tb.Th and Tb.Sp. These include the Local Thickness plugin from ImageJ, the ITKBoneMorphometry module, the PoreSpy library, the SPECTRA collaboration's code, and the deep learning-based segmentation method from Panfilov et al. Each method utilizes different algorithms, such as distance transformation, sphere fitting, and skeleton modeling, to calculate the thickness and separation values.

Trabecular number:
Tb.N is calculated using the formula (BV/TV)/Tb.Th. The open-source tool implements Tb.N calculation based on the computed Tb.Th values from different implementations.

* The script is called "3D_analysis.py,". To execute the procedure from the terminal, the user needs to provide the input folder link and the voxel size as arguments: `python 3D_analysis.py <link to input folder> <voxel_size>`. The output calculations are saved in an Excel file named "3D_analysis_PYTHON" within the input folder. The voxel size is specified in millimeters to allow compatibility with different CT modalities.

* To use different implementations, specific steps must be followed. Running BoneJ headlessly from Python requires downloading ImageJ or Fiji software and installing the BoneJ plugins through Fiji. The script also needs modification, specifically adjusting the `os.environ['JAVA_HOME']` line and the `ij = imagej.init()` line to match the location of the jre file of the Fiji app. PyimageJ wrapper is used to enable BoneJ usage from Python, and installation instructions can be found on the [PyimageJ GitHub page](https://github.com/imagej/pyimagej).

* Porespy and ITKBoneMorphometry methods can be installed using pip by running `pip install porespy` and `pip install itk-bonemorphometry` in the terminal, respectively.

* The ORMIR method from the 2022 ORMIR workshop in Maastricht can be accessed on their [GitHub page](https://github.com/SpectraCollab/ORMIR_XCT). To download and install it using Git Bash, specific commands need to be executed.

* For the MIPT_OULU implementations, the code is available on their [GitHub page](https://github.com/MIPT-Oulu/SubregionalCartilageAnalysis). It can be downloaded and installed by following the provided steps.

Note that the installation steps mentioned above are only required for the initial use of the script.


## Citing
|||
|-----------------------|-----------------|
|**Author / Master student** | Emma Van Riet| 
|**Author / PhD Candidate** | Fahimeh Azari|
|**P.I.**| Harry van Lenthe |
|**Department** | Mechanical Engineering, Biomechanics Section|
|**Institution** | KU Leuven |
  
The future publication cover the technical parts of this project: **link to be added in the near future**

