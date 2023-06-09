import warnings
import cv2
from natsort import natsorted
import glob
import numpy as np
import napari
import SimpleITK as sitk
import os
import sys

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

def save_images(image,output_path_folder):
    """
    Saves the 3D image into 2D bmp files.
    Parameters:
        image (numpy.ndarray): Numpy array 3D image that needs to be saved separately
        in bmp files.
        output_path_folder (str): Directory where the images need to be saved.
    Return:
        void
    """
    filenames = [f for f in os.listdir(path) if
                 os.path.isfile(os.path.join(path, f)) and f.endswith('.bmp')]
    for slice in range(0,image.shape[0]):
        output_path_file = os.path.join(output_path_folder, filenames[slice])
        # Save the images
        image_slice = image[slice,:,:]
        out = np.zeros((image.shape[1], image.shape[2]))
        output_image = cv2.normalize(image_slice, out, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite(output_path_file, output_image)

def adaptive_thresholding_3D(image, radius, thresh):
    """
    Adaptive thresholding: Per slice, for each pixel, the threshold is calculated as
    the midpoint between minimum and maximum greyscales within the square
    kernel with the selected radius. First performs a prethreshold.

    Parameters:
        image (numpy.ndarray): The greyscale 3D input image.
        radius (int): The radius of the square kernel.
        thresh (int): The pre-threshold value.

    Returns:
        numpy.ndarray: The binary 3D image obtained by adaptively thresholding the input image.
    """

    # apply prethreshold
    image = sitk.GetImageFromArray(image)
    image = sitk.Threshold(image,lower=thresh,upper=255,outsideValue=0)
    image = sitk.GetArrayFromImage(image)

    image_padded = np.pad(image, ((radius,radius),
                                  (radius,radius), (radius, radius)), 'constant')
    image_output = np.zeros_like(image)

    # Calculate the mean of the min and max value of the VOI defined by the radius
    for slice in range(radius, image.shape[0]+radius):
        for x in range(radius,image.shape[1]+radius):
            for y in range(radius,image.shape[2]+radius):
                window = image_padded[slice-radius:slice+radius+1,
                         x-radius:x+radius+1, y-radius:y+radius+1]
                threshold = ((window.max() + window.min()) // 2)
                if image[slice-radius, x-radius, y-radius] > threshold:
                    image_output[slice-radius, x-radius, y-radius] = 255

    #view_volume(image_output)

    return image_output.astype(np.uint8)

if __name__ == '__main__':

    try:
        path = sys.argv[1]
        radius = int(sys.argv[2])
        thresh = int(sys.argv[3])

        image_original = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY) for file in
                          natsorted(glob.glob(path + "\*.bmp"))]
        image_original = np.asarray(image_original)
        image_output = adaptive_thresholding_3D(image=image_original,
                                                radius=radius, thresh=thresh)

        output_folder_path = os.path.join(path, '3D_adaptive_thresh')
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        save_images(image_output, output_folder_path)

        print('Thresholded images saved in the subfolder "3D_adaptive_thresh".')

    except:
        if len(sys.argv) != 4:
            warnings.warn('Not the right amount of arguments were given. '
            'As a reminder, the input should be '
            'python Adaptive_Threshold_3D.py <"link to input folder"> <radius> <threshold>')
        else:
            warnings.warn('An error occurred during the thresholding. '
            'Please check the input. As a reminder, the input should be '
            'python Adaptive_Threshold_3D.py <"link to input folder"> <radius> <threshold>')