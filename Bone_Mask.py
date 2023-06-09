import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import (morphological_geodesic_active_contour,
                                 inverse_gaussian_gradient)
import skimage
import cv2
import os
import SimpleITK as sitk
import glob
import napari

def viewVolume(volume):
    """
        Shows the 3D image using the Napari library.
        Parameters:
            volume (numpy.ndarray): Numpy array 3D image.
        Returns:
            void
    """
    viewer = napari.view_image(volume, contrast_limits=[0, 1])
    napari.run()


def store_evolution_in(lst):
   """
   Returns a callback function to store the evolution of the level sets in
   the given list.
   """

   def _store(x):
       lst.append(np.copy(x))

   return _store

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
           image[slice] = cv2.bilateralFilter(image[slice], 9, 0, 1)  # cv2.medianBlur(image_1bone[slice],3)
       image = sitk.GetImageFromArray(image)

    if option == 'median110':
       image = sitk.BinaryMedian(image,(1,1,0))

    if option == 'median111':

       image = sitk.Median(image,(1,1,1))

    if option == 'median221':
       image = sitk.BinaryMedian(image, (2, 2, 1))

    return image

if __name__ == "__main__":
    try:
        bmp_folder_path_PCCT = sys.argv[1]
        thresh_val = int(sys.argv[2])
        kernel_size = 15
        if thresh_val > 110:
            kernel_size = 35

        # getting the filenames for the output
        filenames = [f for f in os.listdir(bmp_folder_path_PCCT) if
                       os.path.isfile(os.path.join(bmp_folder_path_PCCT, f)) and f.endswith('.bmp')]
        output_path_PCCT = os.path.join(bmp_folder_path_PCCT, 'bonemask')
        if not os.path.exists(output_path_PCCT):
           os.makedirs(output_path_PCCT)

        # reading the image
        image = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY) for file in glob.glob(bmp_folder_path_PCCT + "\*.bmp")]
        image_original = np.asarray(image)
        shape_1slice = image_original.shape

        # converting image to SimpleITK format
        image = sitk.GetImageFromArray(image_original)

        # thresholding the image
        image = sitk.Threshold(image,lower=thresh_val,upper=255)
        thresholded_image = sitk.GetArrayFromImage(image)
        image = despeckle(image, 'bilateral901')

        # Despeckle
        image = sitk.ConnectedComponent(image, False)
        image = sitk.RelabelComponent(image, sortByObjectSize=True)
        image = sitk.GetArrayFromImage(image)

        for slice in range(0,shape_1slice[0]):
           print(str(slice))
           image_1slice = image[slice,:,:]
           output_path_file = os.path.join(output_path_PCCT, filenames[slice])
           output_image = np.zeros((shape_1slice[1], shape_1slice[2]), np.uint8)
           for bone in range(1,5):
               one_bone_image_1slice = (image_1slice == bone).astype(np.uint8)

               if thresh_val < 100:
                   one_bone_image_1slice = cv2.bilateralFilter(one_bone_image_1slice, 9, 0, 1)

               empty = np.zeros_like(one_bone_image_1slice)
               one_bone_image_1slice = cv2.normalize(one_bone_image_1slice, empty, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

               if np.mean(one_bone_image_1slice) > 0.0001:

                   # create mask for active contours
                   kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                   trial_image = np.copy(one_bone_image_1slice)
                   mask_VOI = cv2.dilate(trial_image,kernel)
                   mask_VOI = cv2.bitwise_not(mask_VOI)
                   num_labels, labels_im = cv2.connectedComponents(mask_VOI)
                   mask_VOI = (labels_im == 1).astype(np.uint8)
                   empty = np.zeros_like(one_bone_image_1slice)
                   mask_VOI = cv2.normalize(mask_VOI, empty, 0, 255, cv2.NORM_MINMAX,dtype=cv2.CV_8U)
                   mask_VOI = cv2.bitwise_not(mask_VOI)


                   output = thresholded_image[slice,:,:]
                   output = cv2.bitwise_and(output, output, mask=one_bone_image_1slice)
                   image_preprocessed = cv2.bitwise_not(output)
                   image_preprocessed = skimage.img_as_float(image_preprocessed)

                   # Shrink wrap
                   # Initial level set
                   init_ls = mask_VOI

                   evolution = []
                   callback = store_evolution_in(evolution)

                   gimage = inverse_gaussian_gradient(image_preprocessed,alpha=100.0,sigma=4.0)


                   ls = morphological_geodesic_active_contour(gimage,num_iter=80,init_level_set = init_ls,
                                                              smoothing=1,balloon=-1,iter_callback=callback)

                   #code for showing the evolution of the active contours
                   # fig, axes = plt.subplots(1, 2, figsize=(16, 16))
                   # ax = axes.flatten()
                   #
                   # ax[0].imshow(image_original[slice,:,:], cmap="gray")
                   # ax[0].set_axis_off()
                   # ax[0].contour(ls, [0.5], colors='r')
                   # ax[0].set_title("Morphological GAC segmentation", fontsize=12)
                   #
                   # ax[1].imshow(image_preprocessed, cmap="gray") #ls ipv image_preprocessed
                   # ax[1].set_axis_off()
                   # contour = ax[1].contour(evolution[2], [0.5], colors='g')
                   # contour.collections[0].set_label("Iteration 2")
                   # contour = ax[1].contour(evolution[30], [0.5], colors='y')
                   # contour.collections[0].set_label("Iteration 30")
                   # contour = ax[1].contour(evolution[-1], [0.5], colors='r')
                   # contour.collections[0].set_label("Iteration 80")
                   # ax[1].legend(loc="upper right")
                   # title = "Morphological GAC evolution"
                   # ax[1].set_title(title, fontsize=12)
                   # fig.tight_layout()
                   # plt.show()

                   # add the different bones to one image
                   output_image = output_image | ls

           # write image
           out = np.zeros_like(one_bone_image_1slice)
           output_image = cv2.normalize(output_image, out, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
           cv2.imwrite(output_path_file, output_image)

    except:
        if len(sys.argv) != 3:
            warnings.warn('Not the right amount of arguments were given.'
            'As a reminder, the input should be '
            'python Bone_Mask.py <"link to input folder"> <threshold>')
        else:
            warnings.warn('An error occurred during the STB segmentation.'
            'Please check the input. As a reminder, the input should be '
            'python Bone_Mask.py <"link to input folder"> <threshold>')