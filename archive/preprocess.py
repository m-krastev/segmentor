import os

import argparse
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from natsort import natsorted
from tqdm import tqdm

from window_resample import PreprocessWrapper


# DICOM
# reader = sitk.ImageSeriesReader()
# dicom_names = reader.GetGDCMSeriesFileNames("../Data/cervix_001/cervix_002/100000CD")
# reader.SetFileNames(dicom_names)
# reader.MetaDataDictionaryArrayUpdateOn()
# reader.LoadPrivateTagsOn()
# img = reader.Execute()
# size = ct_scan.GetSize()
# print(size)
# sitk.WriteImage(ct_scan, "../Data/test_ct_scan.nii.gz")


def read_nifti_paths(dataset_dir):
    paths = dict()
    for patient in os.listdir(dataset_dir):
        try:
            nifti_paths = os.listdir(f"{dataset_dir}/{patient}/MR_annotated/")
        except (FileNotFoundError, NotADirectoryError):
            continue

        filtered_paths = list(filter(lambda x: x.endswith('.nii'), nifti_paths))
        assert len(filtered_paths) == 1, f"Found more than one match: {filtered_paths}"

        paths[patient] = f"{dataset_dir}/{patient}/MR_annotated/{filtered_paths[0]}"

    return paths


def read_nifti_ann(dataset_dir):
    paths = dict()
    for patient in os.listdir(dataset_dir):
        try:
            nifti_paths = os.listdir(f"{dataset_dir}/{patient}/")
        except (FileNotFoundError, NotADirectoryError):
            continue

        filtered_paths = list(filter(lambda x: x.endswith('midline_delation.nii'), nifti_paths))
        assert len(filtered_paths) == 1, f"Found no exact match for {patient}: {filtered_paths}"

        paths[patient] = f"{dataset_dir}/{patient}/{filtered_paths[0]}"

    return paths

def calc_dist(data, key='img'):
    # Get bins for whole dataset
    all_imgs = [pt[key] for pt in data.values()]
    probs, bins = np.histogram(all_imgs, bins=100)

    # Get probs per image based on bins
    for pt_id, pt_data in data.items():
        counts = np.histogram(pt_data[key], bins=bins)[0]
        data[pt_id]['dist'] = counts / sum(counts)  # Bar heights sum to 1

    return bins, data


def plot(data, bins, bin_start=0, alpha=0.1):
    # Calculate some statistics
    all_probs = [pt['dist'] for pt in data.values()]
    mean_prob = np.mean(all_probs, axis=0)
    median_prob = np.median(all_probs, axis=0)

    # Create barplot per patient
    for pt_id, pt_data in natsorted(data.items()):
        plt.bar(bins[:-1][bin_start:], pt_data['dist'][bin_start:], width=bins[1]*0.9, label=pt_id, alpha=alpha)

    # Create mean and median line plot
    plt.plot(bins[:-1][bin_start:], mean_prob[bin_start:], label="Mean")
    plt.plot(bins[:-1][bin_start:], median_prob[bin_start:], label="Median")

    # Show plot
    plt.title("Distribution of masked MR data")
    plt.xlabel("Intensity value")
    plt.ylabel("Proportion (heights sum to 1)")

    plt.legend()
    plt.show()


def resample_all(path):
    for img in tqdm(os.listdir(path)):
        if not img.endswith('.nii'):
            continue

        new_img = (PreprocessWrapper(path, img).resample((1.25, 1.25, 1.25)))
        sitk.WriteImage(new_img, os.path.join(path, img))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to directory to resample images", required=True)
    args = parser.parse_args()

    resample_all(args.path)

    # img_paths = read_nifti_paths("../Data/MR/imagesTr")  # Read image paths
    # ann_paths = read_nifti_ann("../Data/MR/labelsTr")  # Read annotation paths
    #
    # # Read all images and annotations as arrays
    # data = dict()
    # threshold = 2_500
    # n_voxels, total_voxels = 0, 0
    # for pt in img_paths.keys():
    #     if pt not in ann_paths:
    #         print(f"No annotations found for {pt}")
    #         continue
    #
    #     img = sitk.GetArrayFromImage(sitk.ReadImage(img_paths[pt])).flatten()
    #     ann = sitk.GetArrayFromImage(sitk.ReadImage(ann_paths[pt])).flatten()
    #
    #     masked_img = img * ann
    #
    #     n_voxels += np.sum(img > threshold)
    #     total_voxels += img.size
    #
    #     data[pt] = {'img': img, 'masked_img': masked_img, 'ann': ann, 'dist': []}
    #
    # print(f"#voxels > {threshold}: {n_voxels}, Total #voxels: total_voxels")
    # print(f"Voxels > {threshold}: {round(n_voxels / total_voxels * 100, 4)}%")
    #
    # # Calculate distribution
    # bins, data = calc_dist(data, key='masked_img')
    #
    # # Plot
    # plot(data, bins, bin_start=1, alpha=0.2)
