import numpy as np
import imagej
import sys
import cv2
from natsort import natsorted
import glob
import os
import SimpleITK as sitk
import itk
import porespy
import scyjava as sj
import xlsxwriter
import warnings

# Importing functions from code from ORMIR_XCT_main and ORMIR_XCT_fix-dt-maybe:
from ORMIR_XCT_main.ormir_xct_main.util.hildebrand_thickness import(
    calc_structure_thickness_statistics as main_calc_structure_thickness_statistics)
from SubregionalCartilageAnalysis.scartan.metrics.local_thickness \
    import _local_thickness as local_thickness_MIPT_oulu

os.environ['JAVA_HOME'] = \
    r'C:\Users\Emma Van Riet\Fiji.app\java\win64\zulu8.60.0.21-ca-fx-jdk8.0.322-win_x64\jre '

def Bone_Volume_Fraction_BoneJ(image,ij):
    """
    The bone volume fraction calculated by BoneJ.
    Parameters:
        image (numpy.ndarray): 3D VOI from which the BV/TV will be calculated
        ij: ImageJ gateway
    Returns:
        int: The BV/TV calculated by BoneJ.

    """

    # convert the image to an ImageJ2-style ImgPlus object
    img = ij.py.to_img(image)

    volumefraction = sj.jimport('org.bonej.wrapperPlugins.ElementFractionWrapper')
    sharedtable = sj.jimport('org.bonej.utilities.SharedTable')
    volumefraction_run = ij.command().run(volumefraction, True, ["inputImage",img])
    volumefraction_run.get()
    table = sharedtable.getTable()
    volumefraction = np.mean(table[2])

    return volumefraction

def Trabecular_Thickness_BoneJ(image,ij):
    """
    The trabecular thickness calculated by BoneJ
    Parameters:
        image (numpy.ndarray): 3D VOI from which the Tb.Th will be calculated
        ij: ImageJ gateway
    Returns:
         int: The Tb.Th calculated by BoneJ
    """

    # convert the image to an ImageJ1-style ImagePlus object
    imp = ij.py.to_imageplus(image)
    imp.setDimensions(image.shape[0],image.shape[1],image.shape[2])

    # instantiate local thickness and set its options
    localThickness = sj.jimport('sc.fiji.localThickness.LocalThicknessWrapper')
    localThickness().setSilence(True)
    localThickness().setShowOptions(False)
    localThickness().maskThicknessMap = True
    localThickness().calibratePixels = True

    # run local thickness. mapImp is the thickness map as an ImageJ1 ImagePlus
    mapImp = localThickness().processImage(imp)

    # calculate some summary statistics on the map
    processor = sj.jimport('ij.process.StackStatistics')
    resultStats = processor(mapImp)

    return resultStats.mean

def Trabecular_Thickness_ORMIR_main(image,pixel_size):
    """
    The trabecular thickness calculated by ORMIR
    Parameters:
        image (numpy.ndarray): 3D VOI from which the Tb.Th will be calculated
        pixel_size (int): pixel size of the CT image
    Returns:
         int: The Tb.Th calculated by ORMIR
    """
    thickness_stats = main_calc_structure_thickness_statistics(image,
                            (pixel_size,pixel_size,pixel_size), 0.1)
    return thickness_stats[0]

def Trabecular_Thickness_ITK_bone_morphometry(image):
    """
    The trabecular thickness calculated by ITK_bone_morphometry
    Parameters:
        image (numpy.ndarray): 3D VOI from which the Tb.Th will be calculated
    Returns:
         int: The Tb.Th calculated by ITK_bone_morphometry
    """
    image = image.astype(np.int32)
    filtr = itk.BoneMorphometryFeaturesFilter.New(itk.GetImageFromArray(image).astype(itk.UC))
    filtr.SetMaskImage(itk.GetImageFromArray(image).astype(itk.UC))
    filtr.SetThreshold(1)
    filtr.Update()
    thickness = filtr.GetTbTh()
    return thickness

def Trabecular_Thickness_MIPT_oulu_skeleton(image,pixel_size):
    """
    The trabecular thickness calculated by ORMIR using the skeleton method
    Parameters:
        image (numpy.ndarray): 3D VOI from which the Tb.Th will be calculated
        pixel_size (int): pixel size of the CT image
    Returns:
         int: The Tb.Th calculated by MIPT_OULU using the skeleton method
    """
    thickness = local_thickness_MIPT_oulu(mask=image,mode='straight_skel_3d',
                        spacing_mm=(pixel_size,pixel_size,pixel_size),stack_axis=0)
    actual_thickness = thickness[thickness > 0].mean()
    return actual_thickness

def Trabecular_Thickness_MIPT_oulu_medialaxis(image,pixel_size):
    """
    The trabecular thickness calculated by MIPT_OULU using the medial axis method
    Parameters:
        image (numpy.ndarray): 3D VOI from which the Tb.Th will be calculated
        pixel_size (int): pixel size of the CT image
    Returns:
         int: The Tb.Th calculated by MIPT_OULU using the medial axis method
    """
    thickness = local_thickness_MIPT_oulu(mask=image,mode='med2d_dist3d_lth3d',
                        spacing_mm=(pixel_size,pixel_size,pixel_size),stack_axis=0)
    actual_thickness = thickness[thickness > 0].mean()
    return actual_thickness

def Trabecular_Thickness_Porespy(image):
    """
    The trabecular thickness calculated by PoreSpy
    Parameters:
        image (numpy.ndarray): 3D VOI from which the Tb.Th will be calculated
    Returns:
         int: The Tb.Th calculated by PoreSpy
    """
    thickness = porespy.filters.local_thickness(image,sizes=50,mode='hybrid')
    actual_thickness = thickness[thickness > 0].mean()
    return actual_thickness

def Trabecular_Number(bonevolumefraction,trabecularthickness):
    """
    The trabecular number, calculated using BV/TV and Tb.Th
    Parameters:
        bonevolumefraction (int): BV/TV
        trabecularthickness (int) Tb.Th
    Returns:
         int: The Tb.N
    """
    trabecularnumber = bonevolumefraction/trabecularthickness
    return trabecularnumber


def WriteExcel(image,name_file):
    """
    Calls all the parameter calculation functions and writes the results
    in an Excel file.
    Parameters:
        image (numpy.ndarray): 3D VOI from which the parameters will be calculated
        name_file (str): name of the Excel file
    Returns:
         void
    """

    workbook = xlsxwriter.Workbook(name_file)
    worksheet = workbook.add_worksheet()

    worksheet.write('B1', 'BoneJ')
    worksheet.write('C1', 'ORMIR-main')
    worksheet.write('D1', 'ITK_bone_morphometry')
    worksheet.write('E1', 'MIPT_oulu_skeleton')
    worksheet.write('F1', 'MIPT_oulu_medialaxis')
    worksheet.write('G1', 'Porespy')

    worksheet.write('A2', 'Tb.Th')
    worksheet.write('A3', 'Tb.Sp')
    worksheet.write('A4', 'BV/TV')
    worksheet.write('A5', 'Tb.N')

    # Write the trabecular thickness values
    TbTh_boneJ = Trabecular_Thickness_BoneJ(image,ij=ij)*pixel_size
    worksheet.write('B2', TbTh_boneJ)
    TbTh_ORMIR = Trabecular_Thickness_ORMIR_main(image,pixel_size=pixel_size)
    worksheet.write('C2',TbTh_ORMIR)
    TbTh_ITK = Trabecular_Thickness_ITK_bone_morphometry(image)*pixel_size
    worksheet.write('D2',TbTh_ITK)
    TbTh_MIPT_skel = Trabecular_Thickness_MIPT_oulu_skeleton(image,pixel_size=pixel_size)
    worksheet.write('E2',TbTh_MIPT_skel)
    TbTh_MIPT_medial = Trabecular_Thickness_MIPT_oulu_medialaxis(image,pixel_size=pixel_size)
    worksheet.write('F2',TbTh_MIPT_medial)
    TbTh_Porespy = Trabecular_Thickness_Porespy(image)*pixel_size
    worksheet.write('G2', TbTh_Porespy)

    # Write bone volume fraction from BoneJ
    BVTV_boneJ = Bone_Volume_Fraction_BoneJ(image, ij=ij)
    worksheet.write('B4', BVTV_boneJ)

    # Write trabecular number from BoneJ
    Tb_N_boneJ = Trabecular_Number(BVTV_boneJ,TbTh_boneJ*pixel_size)
    worksheet.write('B5',Tb_N_boneJ)
    Tb_N_ORMIR = Trabecular_Number(BVTV_boneJ,TbTh_ORMIR)
    worksheet.write('C5',Tb_N_ORMIR)
    Tb_N_ITK = Trabecular_Number(BVTV_boneJ,TbTh_ITK)
    worksheet.write('D5',Tb_N_ITK)
    Tb_N_MIPT_skel = Trabecular_Number(BVTV_boneJ,TbTh_MIPT_skel)
    worksheet.write('E5',Tb_N_MIPT_skel)
    Tb_N_MIPT_medial = Trabecular_Number(BVTV_boneJ,TbTh_MIPT_medial)
    worksheet.write('F5',Tb_N_MIPT_medial)
    Tb_N_Porespy = Trabecular_Number(BVTV_boneJ,TbTh_Porespy)
    worksheet.write('G5',Tb_N_Porespy)

    # Write the trabecular separation values
    image = sitk.GetImageFromArray(image)
    image = sitk.Cast(sitk.IntensityWindowing(image, windowMinimum=0,
                windowMaximum=254, outputMinimum=0.0, outputMaximum=1.0), sitk.sitkUInt8)
    image = sitk.BinaryNot(image)
    image = sitk.Cast(sitk.IntensityWindowing(image, windowMinimum=0,
                windowMaximum=1, outputMinimum=0.0, outputMaximum=254), sitk.sitkUInt8)
    image = sitk.GetArrayFromImage(image)

    TbSp_boneJ = Trabecular_Thickness_BoneJ(image,ij=ij)*pixel_size
    worksheet.write('B3', TbSp_boneJ)
    TbSp_ORMIR = Trabecular_Thickness_ORMIR_main(image,pixel_size=pixel_size)
    worksheet.write('C3', TbSp_ORMIR)
    TbSp_ITK = Trabecular_Thickness_ITK_bone_morphometry(image)*pixel_size
    worksheet.write('D3', TbSp_ITK)
    TbSp_MIPT_skel = Trabecular_Thickness_MIPT_oulu_skeleton(image,pixel_size=pixel_size)
    worksheet.write('E3', TbSp_MIPT_skel)
    TbSp_MIPT_medial = Trabecular_Thickness_MIPT_oulu_medialaxis(image,pixel_size=pixel_size)
    worksheet.write('F3', TbSp_MIPT_medial)
    TbSp_Porespy = Trabecular_Thickness_Porespy(image)*pixel_size
    worksheet.write('G3', TbSp_Porespy)

    workbook.close()

if __name__ == '__main__':

    try:
        path = sys.argv[1]
        pixel_size = float(sys.argv[2])

        ij = imagej.init(r'C:\Users\Emma Van Riet\Fiji.app', mode=imagej.Mode.HEADLESS)
        print(f"ImageJ2 version: {ij.getVersion()}")

        image_original = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
                          for file in natsorted(glob.glob(path + "\*.bmp"))]
        image_original = np.asarray(image_original)

        excel_file = '3D_analysis_PYTHON.xlsx'
        directory_excel = os.path.join(path,excel_file)
        WriteExcel(image_original,name_file=directory_excel)

        print('Excel file with calculated parameters saved '
              'in the subfolder "3D_analysis_PYTHON.xlsx".')

    except:
        if len(sys.argv) != 3:
            warnings.warn('Not the right amount of arguments were given. '
            'As a reminder, the input should be '
            'python 3D_Analysis.py <"link to input folder"> <voxel_size>')
        else:
            warnings.warn('An error occurred during the thresholding. Please check the input.'
                          'As a reminder, the input should be '
                          'python 3D_Analysis.py <"link to input folder"> <voxel_size>')