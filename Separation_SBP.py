import SimpleITK as sitk
import numpy as np
import cv2
import glob
import gc
import sys
import os
from natsort import natsorted
import napari
import matplotlib.pyplot as plt
import warnings

def view_volume(volume):
    """
    Shows the 3D image using the Napari library.
    Parameters:
        volume (numpy.ndarray): Numpy array 3D image.
    Returns:
        void
    """
    viewer = napari.view_image(volume, contrast_limits=[0, 1])
    napari.run()

def viewSlice(slice):
    """
    Shows the 2D image using the matplotlib library.
    Parameters:
        slice (numpy.ndarray): Numpy array 2D image.
    Returns:
        void
    """
    plt.imshow(slice, interpolation='nearest')
    plt.gray()
    plt.show()

def save_images(image,output_path_folder):
    """
    Saves the 3D image into 2D bmp files.
    Parameters:
    image (numpy.ndarray): Numpy array 3D image that needs to be saved separately
    in bmp files
    output_path_folder (str): Directory where the images need to be saved
    Returns:
        void
    """
    filenames = [f for f in os.listdir(bmp_folder_path) if
               os.path.isfile(os.path.join(bmp_folder_path, f)) and f.endswith('.bmp')]

    image_slice_empty = np.zeros((image.shape[1], image.shape[2]))

    for slice in range(0, image_original.shape[0]):
        output_path_file = os.path.join(output_path_folder, filenames[slice])

        if slice < slice_number1:
            output_image = image_slice_empty
        elif slice_number1 <= slice < slice_number2:
            # Save the images
            image_slice = image[slice-slice_number1, :, :]
            output_image = cv2.normalize(image_slice, image_slice_empty, 0, 255,
                                         cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            output_image = image_slice_empty

        cv2.imwrite(output_path_file, output_image)


def process_slice(slice_index,imageVOI_slice,radius):
   """
      Adaptive thresholding: Per slice, for each pixel, the threshold is calculated as
      the midpoint between minimum and maximum greyscales within the square
      kernel with the selected radius. First performs a prethreshold.

      Parameters:
          slice_index (int): The number of the slice being thresholded
          imageVOI_slice (numpy.ndarray): The greyscale 2D input slice
          radius (int): The radius of the square kernel
      Returns:
          numpy.ndarray: The binary 2D slice obtained by thresholding the input image
      """
   print(str(slice_index))
   slice_image = image_original[slice_index]
   slice_padded = image_padded[slice_index]
   slice_output = np.zeros_like(slice_image)

   bboxx1,bboxy1, width, height = cv2.boundingRect(imageVOI_slice)
   bboxx2 = bboxx1 + width
   bboxy2 = bboxy1 + height

   if bboxy1-30 < radius: bboxy1 = radius
   else: bboxy1 = bboxy1-30
   if bboxy2+30 > slice_padded.shape[0]-radius:
       bboxy2 = slice_padded.shape[0]-radius
   else: bboxy2 = bboxy2+30
   if bboxx1-30 < radius: bboxx1 = radius
   else: bboxx1 = bboxx1-30
   if bboxx2+30 > slice_padded.shape[1]-radius:
       bboxx2 = slice_padded.shape[1]-radius
   else: bboxx2 = bboxx2+30

   # Iterate over each pixel
   for x in range(bboxy1, bboxy2):
       for y in range(bboxx1, bboxx2):
           # Get the local region
           local_region = slice_padded[x-radius:x+radius+1, y-radius:y+radius+1]

           max_local = local_region.max()
           if max_local != 0:
               min_local = local_region.min()

               # Compute the local midpoint of the min and max value
               threshold = (max_local + min_local) // 2

               # Threshold the center pixel
               if slice_image[x-radius, y-radius] > threshold:
                   slice_output[x-radius, y-radius] = 255

   return slice_output

def despeckle(image,option):
    """
    Despeckles the 3D SimpleITK image using one of the options.
    The options are 'bilateral901', 'median110', 'median111'
    and 'median221'.
    Parameters:
        image (sitk.Image): 3D image that needs to be despeckled
        option: despeckling option
    Returns
        sitk.Image: The despeckled image
    """

    if option == 'bilateral901':
       image = sitk.GetArrayFromImage(image)
       for slice in range(0, image.shape[0]):
           image[slice] = cv2.bilateralFilter(image[slice], 9, 0, 1)
       image = sitk.GetImageFromArray(image)

    if option == 'median110':
       image = sitk.BinaryMedian(image,(1,1,0))

    if option == 'median111':
       image = sitk.BinaryMedian(image,(1,1,1))

    if option == 'median221':
       image = sitk.BinaryMedian(image, (2, 2, 1))

    return image


def Separate_1bone(image,bone=None):
    """
    Separates a specific bone, the tibia or the femur.
    parameters:
        image (sitk.Image): 3D binary image from which one bone is separated
        bone (str): string stating which bone should be segmented.
        Can be 'tibia' or 'femur'
    Returns:
        A 3D binary SimpleITK image containing only the requested bone
    """
    if bone == 'tibia':
       bone = 2
    elif bone == 'femur':
       bone = 1

    # Connectivity filter to separate and label the different objects
    image = sitk.ConnectedComponent(image, False)
    image = sitk.RelabelComponent(image, sortByObjectSize=True)
    image_connected_bones = sitk.GetArrayFromImage(image)

    # Separate the required bone
    image = image_connected_bones == bone
    image = image.astype('uint8')

    list_slice = np.sum(image, axis=(-1, -2)) != 0
    image = image[list_slice]

    # if threshold not high enough to separate the bones
    while image.shape == image_original.shape:
       image = sitk.GetImageFromArray(image)
       image = despeckle(image, 'bilateral901')
       image = despeckle(image, 'median110')
       image = sitk.ConnectedComponent(image, False)
       image = sitk.RelabelComponent(image, sortByObjectSize=True)
       image_connected_bones = sitk.GetArrayFromImage(image)
       image = image_connected_bones == bone
       image = image.astype('uint8')

       list_slice = np.sum(image, axis=(-1, -2)) != 0
       image = image[list_slice]
    global slice_number1
    global slice_number2
    slice_number1 = np.where(list_slice == True)[0][0]
    slice_number2 = np.where(list_slice == True)[0][-1]

    image = sitk.GetImageFromArray(image)

    return image

def Closing_with_connectivity(image_original,kernel_size):
    """
    Performs a morphological closing operation with a connectivity
    filter in between the dilation and the erosion to fill the holes.
    Parameters:
        image_original (sitk.Image): 3D SimpleITK image
        kernel_size (int): kernels size for the dilation and erosion
    Returns:
        sitk.Image: The 3D image which has undergone the closing with the
        connectivity filter

    """
    # Dilate the image
    image = sitk.BinaryDilate(image_original,
                              (kernel_size, kernel_size, 0), sitk.sitkBall)

    # 3D connectivity filter on background to get mask
    image = sitk.BinaryNot(image, 0, 1)
    image = sitk.ConnectedComponent(image, True)
    image = sitk.RelabelComponent(image, sortByObjectSize=True)
    image = sitk.GetArrayFromImage(image)
    image = (image == 1)
    image = image.astype('uint8')
    image = sitk.GetImageFromArray(image)
    image = sitk.BinaryNot(image, 1, 0)

    # Erode the image
    image = sitk.BinaryErode(image, (kernel_size, kernel_size, 0), sitk.sitkBall)

    return image

def PerioMask(image_thresholded,bone,kernel):
    """
    Creates a periosteal mask of the greyscale input image.
    Parameters:
        image_thresholded (numpy.ndarray): A 3D greyscale numpy array image
        bone (str): String specifying which bone needs to be separated
        kernel: kernel size of the closing operation
    Returns:
        - numpy.ndarray: The periomask
        - numpy.ndarray: The separated bone
    """
    max_value = int(image_thresholded.max())
    image_thresholded = sitk.GetImageFromArray(image_thresholded)
    image_thresholded = sitk.Cast(sitk.IntensityWindowing(
        image_thresholded, windowMinimum=0.0, windowMaximum=max_value,
        outputMinimum=0.0, outputMaximum=1.0), sitk.sitkUInt8)

    # Separate the desired bone
    print('getting one bone')
    image_1bone = Separate_1bone(image_thresholded,bone)

    # Morphological closing with connectivity
    periomask = Closing_with_connectivity(image_1bone,kernel)

    periomask = sitk.Or(periomask, image_1bone)

    return periomask, image_1bone

def STBmask(image_thresholded_1bone,periomask,kernel):
    """
    Creates the STB mask of the bone.
    Parameters:
        image_thresholded_1bone (sitk.Image): Thresholded image with separated bone
        periomask (sitk.Image): The periomask
        kernel (tuple): The three kernel sizes: for the closing, the dilation and
        the erosion
        Returns:
            numpy.ndarray: The STB mask of the bone
    """
    print('creating endomask')
    image_thresholded_1bone = despeckle(image_thresholded_1bone,'bilateral901')

    # Erode the periomask to remove the cortical shell
    periomask_eroded = sitk.BinaryErode(periomask,
                                (kernel[2], kernel[2], 0), sitk.sitkBall)

    # Subtract the eroded periomask from the original one
    periomask_actual = sitk.Subtract(periomask, periomask_eroded)

    # Dilate the thresholded image to create a more continuous cortical shell
    image_dilated = sitk.BinaryDilate(image_thresholded_1bone,
                                (kernel[1], kernel[1], 0), sitk.sitkBall)

    # Subtract the thresholded image from the periomask
    image_masked = sitk.Subtract(periomask, image_dilated)
    image_masked = image_masked == 1

    # Subtract the actual periomask from the thresholded image
    endomask = sitk.Subtract(image_masked,periomask_actual)
    endomask = endomask == 1

    # Closing operation on the image
    endomask = Closing_with_connectivity(endomask,kernel_size=kernel[0])

    periomask_actual = sitk.Subtract(periomask, endomask)
    periomask_actual = periomask_actual == 1

    return endomask, periomask_actual

if __name__ == "__main__":

    #try:
        bmp_folder_path = sys.argv[1]
        bone = sys.argv[2]
        radius = int(sys.argv[3])
        kernel = eval(sys.argv[4])
        thresh = int(sys.argv[5])

        # Read in image
        image_original = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
                for file in natsorted(glob.glob(bmp_folder_path + "\*.bmp"))]
        image_original = np.asarray(image_original)

        slice_number1 = 0
        slice_number2 = 0

        # Convert the input array to SimpleITK image
        image_original = sitk.GetImageFromArray(image_original)

        # Threshold the image using SimpleITK
        image_original = sitk.Threshold(image_original, lower=thresh,
                                        upper=255, outsideValue=0)

        # Get VOIS
        image_VOI = sitk.ConnectedComponent(image_original, False)
        image_VOI = sitk.RelabelComponent(image_VOI, sortByObjectSize=True)
        image_VOI = sitk.GetArrayFromImage(image_VOI)
        image_VOI = (image_VOI == 1) + (image_VOI == 2)
        image_VOI = image_VOI.astype('uint8')
        image_VOI = np.pad(image_VOI, ((0,0),(radius,radius),
                                       (radius,radius)), 'constant')

        # Convert the thresholded image back to NumPy array
        image_original = sitk.GetArrayFromImage(image_original)

        # Pad the image
        image_padded = np.pad(image_original, ((0, 0), (radius, radius),
                                               (radius, radius)), 'constant')

        # Create an output array
        image_output = np.zeros_like(image_original)

        # Perform adaptive thresholding
        for slice in range(0,image_original.shape[0]):
           image_VOI_slice = image_VOI[slice]
           slice_output = process_slice(slice,image_VOI_slice,radius=radius)
           image_output[slice] = slice_output

        del image_padded
        del image_VOI
        gc.collect()

        image_output = image_original

        if bone == 'both':

           # femur
           bone = 'femur'
           periomask, image_1bone = PerioMask(image_thresholded=image_output,
                                              bone=bone, kernel=kernel[0])
           endomask, periomask_actual = STBmask(image_thresholded_1bone=image_1bone,
                                                periomask=periomask, kernel=kernel)

           # Saving the images
           endomask = sitk.GetArrayFromImage(endomask)
           whole_mask = sitk.GetArrayFromImage(periomask)
           output_path_folder_endo = os.path.join(bmp_folder_path,'femur_STB')
           output_path_folder_mask = os.path.join(bmp_folder_path,'femur_wholemask')
           if not os.path.exists(output_path_folder_endo):
               os.makedirs(output_path_folder_endo)
           if not os.path.exists(output_path_folder_mask):
               os.makedirs(output_path_folder_mask)
           save_images(endomask, output_path_folder=output_path_folder_endo)
           save_images(whole_mask, output_path_folder=output_path_folder_mask)

           # Save memory by deleting no longer used variables
           del endomask
           del periomask_actual
           del periomask
           del image_1bone
           gc.collect()

           # tibia
           bone = 'tibia'
           periomask, image_1bone = PerioMask(image_thresholded=image_output,
                                              bone=bone, kernel=kernel[0])
           endomask, periomask_actual = STBmask(image_thresholded_1bone=image_1bone,
                                                periomask=periomask, kernel=kernel)

           # Saving the images
           endomask = sitk.GetArrayFromImage(endomask)
           whole_mask = sitk.GetArrayFromImage(periomask)
           output_path_folder_endo = os.path.join(bmp_folder_path, 'tibia_STB')
           output_path_folder_mask = os.path.join(bmp_folder_path, 'tibia_wholemask')
           if not os.path.exists(output_path_folder_endo):
               os.makedirs(output_path_folder_endo)
           if not os.path.exists(output_path_folder_mask):
               os.makedirs(output_path_folder_mask)
           save_images(endomask, output_path_folder=output_path_folder_endo)
           save_images(whole_mask, output_path_folder=output_path_folder_mask)

        elif bone == 'femur':
           # femur
           bone = 'femur'
           periomask, image_1bone = PerioMask(image_thresholded=image_output,
                                              bone=bone, kernel=kernel[0])
           endomask, periomask_actual = STBmask(image_thresholded_1bone=image_1bone,
                                                periomask=periomask, kernel=kernel)

           # Saving the images
           endomask = sitk.GetArrayFromImage(endomask)
           whole_mask = sitk.GetArrayFromImage(periomask)
           output_path_folder_endo = os.path.join(bmp_folder_path, 'femur_STB')
           output_path_folder_mask = os.path.join(bmp_folder_path, 'femur_wholemask')
           if not os.path.exists(output_path_folder_endo):
               os.makedirs(output_path_folder_endo)
           if not os.path.exists(output_path_folder_mask):
               os.makedirs(output_path_folder_mask)
           save_images(endomask, output_path_folder=output_path_folder_endo)
           save_images(whole_mask, output_path_folder=output_path_folder_mask)

        elif bone == 'tibia':
           # tibia
           bone = 'tibia'
           #Extra despeckling
           image_output = sitk.GetImageFromArray(image_output)
           image_output = despeckle(image_output, 'bilateral901')
           image_output = sitk.Cast(sitk.IntensityWindowing(
               image_output, windowMinimum=0.0, windowMaximum=255.0,
               outputMinimum=0.0, outputMaximum=1.0),
               sitk.sitkUInt8)
           image_output = despeckle(image_output, 'median111')
           image_output = sitk.Cast(sitk.IntensityWindowing(
               image_output, windowMinimum=0.0, windowMaximum=1.0,
               outputMinimum=0.0, outputMaximum=255.0),
               sitk.sitkUInt8)
           image_output = sitk.GetArrayFromImage(image_output)

           periomask, image_1bone = PerioMask(image_thresholded=image_output,
                                              bone=bone, kernel=kernel[0])
           endomask, periomask_actual = STBmask(image_thresholded_1bone=image_1bone,
                                                periomask=periomask, kernel=kernel)

           # Saving the images
           endomask = sitk.GetArrayFromImage(endomask)
           whole_mask = sitk.GetArrayFromImage(periomask)
           output_path_folder_endo = os.path.join(bmp_folder_path, 'tibia_STB')
           output_path_folder_mask = os.path.join(bmp_folder_path, 'tibia_wholemask')
           if not os.path.exists(output_path_folder_endo):
               os.makedirs(output_path_folder_endo)
           if not os.path.exists(output_path_folder_mask):
               os.makedirs(output_path_folder_mask)
           save_images(endomask, output_path_folder=output_path_folder_endo)
           save_images(whole_mask, output_path_folder=output_path_folder_mask)
    #except:
        if len(sys.argv) != 6:
            warnings.warn('Not the right amount of arguments were given. '
            'As a reminder, the input should be '
            'python Separation_STB.py <"link to input folder"> <"bone"> '
                          '<radius> <"kernel sizes"> <threshold>')
        else:
            warnings.warn('An error occurred during the STB segmentation. '
            'Please check the input. As a reminder, the input should be '
            'python Separation_STB.py <"link to input folder"> <"bone"> '
                          '<radius> <"kernel sizes"> <threshold>')
