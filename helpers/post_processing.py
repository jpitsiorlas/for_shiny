import SimpleITK as sitk
import numpy as np

#--------------------------- Post-processing code -----------------------------------

def run_post_processing(pred_img,kernel_size, conn_comp):

    pred_img = pred_img.numpy()
    sitk_pred = sitk.GetImageFromArray(pred_img)
    sitk_pred = sitk.Cast(sitk_pred, sitk.sitkUInt8)

    #Dilate
    dilate_filter = sitk.BinaryDilateImageFilter()
    dilate_filter.SetKernelType(sitk.sitkBall)
    dilate_filter.SetKernelRadius( kernel_size )
    dil_img = dilate_filter.Execute( sitk_pred )

    #Get largest connected component
    component_image = sitk.ConnectedComponent( dil_img )
    sorted_component_image = sitk.RelabelComponent(component_image, sortByObjectSize=True)
    largest_component_binary_image = sorted_component_image == conn_comp

    #Erode with the same kernel to restore the image
    erode_filter = sitk.BinaryErodeImageFilter()
    erode_filter.SetKernelType(sitk.sitkBall)
    erode_filter.SetKernelRadius ( kernel_size )
    segmentation = erode_filter.Execute ( largest_component_binary_image )

    seg_numpy = sitk.GetArrayFromImage(segmentation)

    return seg_numpy
