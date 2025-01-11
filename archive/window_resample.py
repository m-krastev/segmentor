import numpy as np
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt


class PreprocessWrapper:
    '''
    This class preprocess a CT scan in nifti format, using simpleitk library.
    Input: the folder path of the nifti files, the name os the scan
    functions: return the original spacing and size; windowing the scan, resampling the scan, show the voxel intensity
               distribution, save the processed scan in the given path
    '''

    def __init__(self, folder_path, nifti_name):
        self.img_itk = sitk.ReadImage(os.path.join(folder_path, nifti_name))
        self.img_spacing = self.img_itk.GetSpacing()
        self.img_size = self.img_itk.GetSize()
        self.img_direction = self.img_itk.GetDirection()
        self.img_origin = self.img_itk.GetOrigin()

    def resample(self, new_spacing, interpolator=sitk.sitkNearestNeighbor):
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetInterpolator(interpolator)
        resample_filter.SetOutputDirection(self.img_direction)
        resample_filter.SetOutputOrigin(self.img_origin)
        resample_filter.SetOutputSpacing(new_spacing)
        new_size = np.ceil(np.array(self.img_size) * np.array(self.img_spacing) / new_spacing)
        resample_filter.SetSize([int(new_size[0]), int(new_size[1]), int(new_size[2])])
        resampled_img = resample_filter.Execute(self.img_itk)
        return resampled_img

    def windowing(self, resampled_img, window_size, output_intensity):
        window_filter = sitk.IntensityWindowingImageFilter()
        window_filter.SetWindowMinimum(window_size[0])
        window_filter.SetWindowMaximum(window_size[1])
        window_filter.SetOutputMinimum(output_intensity[0])
        window_filter.SetWindowMaximum(output_intensity[1])
        windowed_img = window_filter.Execute(resampled_img)
        if output_intensity[0] >= 0 and output_intensity[1] < 256:
            # 8 bit int ranges from 0-256, in this format to minimize the img size
            cast_filter = sitk.CastImageFilter()
            cast_filter.SetOutputPixelType(sitk.sitkUInt8)
            windowed_img = cast_filter.Execute(windowed_img)
        return windowed_img

    def histogram_plot(self, img_itk):
        img_arr = np.array(sitk.GetArrayFromImage(img_itk))
        img_flat = img_arr.flatten()
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(img_arr[0, :, :], cmap='gray')
        axs[0].set_title('Image')
        axs[1].hist(img_flat, bins=30)
        axs[1].set_title('Histogram')
        plt.show()
