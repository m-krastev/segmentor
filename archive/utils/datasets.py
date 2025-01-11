import os
import random
import typing
from collections import defaultdict
from copy import copy, deepcopy
from typing import Any

import matplotlib.pyplot as plt
import torchio as tio
import numpy as np
import scipy
import skfmm
import torch
from torch import Tensor
from torch.utils.data import Dataset
import SimpleITK as sitk
from torchvision.transforms.v2 import functional as v2F
from torchvision import transforms


def get_sdf(label, normalize=False):
    # Get inside of label volume
    struct = scipy.ndimage.generate_binary_structure(3, 1)
    inside = scipy.ndimage.binary_erosion(label.bool()[0], structure=struct, iterations=1)

    label = label.int()
    label = 1 - label  # Change outside 0 -> 1 & inside 1 -> 0

    label[0][inside] = -1  # Change inside 0 -> -1, but leave surface at 0
    sdf_label = torch.from_numpy(skfmm.distance(label).astype(np.float32))

    if normalize:
        sdf_label = sdf_label / torch.abs(sdf_label).max()  # Zero stays at zero

    # Inverse variant
    # sdf_label = torch.from_numpy(skfmm.distance(label).astype(np.float32))
    # sdf_label = sdf_label - sdf_label.max()  # TODO: temp inverse
    # sdf_label = abs(sdf_label)

    return sdf_label


class MRImageDataset(Dataset):
    # For reproducibility, fix dataset split
    BowelCenter1_REPRODUCE_PATIENTS1 = [10, 22, 18, 15, 8, 1, 6, 16, 9, 20, 7, 13, 3]
    BowelCenter1_REPRODUCE_PATIENTS2 = [19, 11, 12, 5]
    BowelSeg_REPRODUCE_PATIENTS1 = [13, 22, 16, 1, 11, 23, 3, 5, 9, 6, 8, 20, 19, 15, 18, 17]
    BowelSeg_REPRODUCE_PATIENTS2 = [10, 12, 14, 7]

    def __init__(
            self,
            use_transforms: bool,
            args: dict,
            img_dir: str = None,
            ann_dir: str = None,
            inference: bool = False,
            use_sdf: bool = True,
    ):
        """Create the dataset. Handles both training and testing data.

        Args:
        """
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.use_transforms = False if (args['data']['sanity_check'] or inference) else use_transforms
        self.use_sdf = use_sdf
        self.inference = inference
        self.args = args

        self.repr_pts1 = self.BowelSeg_REPRODUCE_PATIENTS1 if "Segmentation" in args['data']['train']['img_dir'] else self.BowelCenter1_REPRODUCE_PATIENTS1
        self.repr_pts2 = self.BowelSeg_REPRODUCE_PATIENTS2 if "Segmentation" in args['data']['train']['img_dir'] else self.BowelCenter1_REPRODUCE_PATIENTS2

        self.imgs = []
        self.meta = []
        self.labels = []
        self.sdf_labels = []

        # Data augmentations
        self.rotations = range(0, int(self.args['data']['augmentations']['rotate_max_deg']) + 1, 5)
        self.elastic = transforms.ElasticTransform()
        self.gamma_transform = tio.RandomGamma(log_gamma=(-0.3, 0.3))
        self.biasfield_transform = tio.transforms.RandomBiasField()

        if self.img_dir:
            self.read_data()

        # Use only one image for sanity checking per dataset: 1 train, 1 val
        if self.args['data']['sanity_check']:
            idx = random.randrange(0, len(self.imgs)-1)  # Pick random index
            print(f"SANITY CHECKING WITH IDX {idx} to {idx+1}")
            self.imgs = self.imgs[idx:idx+2]
            self.labels = self.labels[idx:idx+2]
            self.sdf_labels = self.sdf_labels[idx:idx+2]
            self.meta = self.meta[idx:idx+2]

    def read_data(self):
        # Read the data
        for img_name in os.listdir(self.img_dir)[:2000]:
            if not img_name.endswith('.nii'):
                continue

            img = sitk.ReadImage(f"{self.img_dir}/{img_name}")
            spacing, origin, direction = torch.tensor(img.GetSpacing()), torch.tensor(img.GetOrigin()), torch.tensor(img.GetDirection())

            img = torch.from_numpy(sitk.GetArrayFromImage(img))[None, :]
            self.imgs.append(img)
            self.meta.append({"spacing": spacing, 'origin': origin, 'direction': direction, 'name': img_name.replace('.nii', '')})

            if self.ann_dir:
                label = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(f"{self.ann_dir}/{img_name}")).astype(np.uint8))[None, :]
            else:
                label = None

            self.labels.append(label)

            if self.use_sdf:
                sdf_label = get_sdf(label, normalize=self.args['data'].get('normalise_sdf', False))
                self.sdf_labels.append(sdf_label)

    def __len__(self):
        return len(self.imgs)

    def create_split(self, split_frac=0.8):
        # Create mapping from patient to sample idx
        per_patient = defaultdict(list)
        for idx, info in enumerate(self.meta):
            patient_id = int(info['name'].split('_')[1])
            per_patient[patient_id].append(idx)

        # Split based on patient
        all_patients = list(per_patient.keys())
        patients_1 = random.sample(all_patients, int(len(all_patients) * split_frac))
        patients_2 = list(set(all_patients) - set(patients_1))

        assert self.args['data']['sanity_check'] or (len(patients_1) == len(self.repr_pts1)), f"Lengths not correct got {len(self.repr_pts1)}, expected {len(patients_1)}. Probably training BowelSeg insetead of BowelCenter1"

        # TODO: read this from args, because hardcoded now for reproducibility
        patients_1 = self.repr_pts1 if not self.args['data']['sanity_check'] else patients_1
        patients_2 = self.repr_pts2 if not self.args['data']['sanity_check'] else patients_2

        og_imgs, og_labels, og_sdf_labels, og_meta = copy(self.imgs), copy(self.labels), copy(self.sdf_labels), copy(self.meta)

        # Change current dataset
        self.imgs = [og_imgs[img_idx] for patient_idx in patients_1 for img_idx in per_patient[patient_idx]]
        self.labels = [og_labels[img_idx] for patient_idx in patients_1 for img_idx in per_patient[patient_idx]]
        if self.use_sdf:
            self.sdf_labels = [og_sdf_labels[img_idx] for patient_idx in patients_1 for img_idx in per_patient[patient_idx]]
        self.meta = [og_meta[img_idx] for patient_idx in patients_1 for img_idx in per_patient[patient_idx]]

        # Switch off sanity check flag for new dataset, because it will be too small
        temp_args = deepcopy(self.args)
        temp_args['data']['sanity_check'] = False

        # Create new dataset
        new_dataset = MRImageDataset(use_transforms=self.use_transforms, args=temp_args, inference=self.inference, use_sdf=self.use_sdf)
        new_dataset.imgs = [og_imgs[img_idx] for patient_idx in patients_2 for img_idx in per_patient[patient_idx]]
        new_dataset.labels = [og_labels[img_idx] for patient_idx in patients_2 for img_idx in per_patient[patient_idx]]
        if self.use_sdf:
            new_dataset.sdf_labels = [og_sdf_labels[img_idx] for patient_idx in patients_2 for img_idx in per_patient[patient_idx]]
        new_dataset.meta = [og_meta[img_idx] for patient_idx in patients_2 for img_idx in per_patient[patient_idx]]

        return new_dataset

    def transforms(self, img, label=None, sdf_label=None):
        # Noise or blur
        if random.random() < self.args['data']['augmentations']['blur_prob']:
            img = v2F.gaussian_blur(img, kernel_size=[3, 3])
        elif random.random() < self.args['data']['augmentations']['noise_prob']:
            sigma = torch.sqrt(torch.std(img)) * 1.5
            img = img + (sigma * torch.randn_like(img))

        # Hflip
        if random.random() < self.args['data']['augmentations']['hflip_prob']:
            img = v2F.hflip(img)
            if label is not None:
                label = v2F.hflip(label)
            if sdf_label is not None:
                sdf_label = v2F.hflip(sdf_label)

        # Rotate
        deg = random.sample(self.rotations, 1)[0]
        img = v2F.rotate(img, deg, interpolation=2)  # smoother interpolation for MR data
        label = v2F.rotate(label, deg)  # interpolation should be kept at nearest neighbour to keep binary labels
        if sdf_label is not None:
            sdf_label = v2F.rotate(sdf_label, deg, interpolation=2)

        # Gamma
        if random.random() < self.args['data']['augmentations']['gamma_prob']:
            img = self.gamma_transform(img)

        # BiasField
        if random.random() < self.args['data']['augmentations']['biasfield_prob']:
            img = self.biasfield_transform(img)

        # Elastic
        if random.random() < self.args['data']['augmentations']['elastic_prob']:
            _, _, height, width = img.shape
            displacement = self.elastic.get_params(self.elastic.alpha, self.elastic.sigma, [height, width])
            img = v2F.elastic(img, displacement, self.elastic.interpolation, self.elastic.fill)
            label = v2F.elastic(label, displacement, 0, self.elastic.fill)  # Use nearest neighbour interpolation for the label to keep it binary
            # Need to overwrite SDF label because label changed
            if sdf_label is not None:
                sdf_label = get_sdf(label, normalize=self.args['data'].get('normalise_sdf', False))

        if sdf_label is not None:
            return img, label, sdf_label
        return img, label

    def __getitem__(self, idx: int) -> tuple[Any, Tensor, Any] | tuple[Any, Tensor]:
        """Get the image and label at the given index.
        Args:
            idx (int): index of the image to get.

        Returns:
            typing.Union[torch.Tensor, torch.Tensor]: image (channels, width, height) and label (1/2, channels, width, height)
        """
        image = self.imgs[idx]
        label = self.labels[idx]

        if self.use_sdf and label is not None:
            sdf_label = self.sdf_labels[idx]

        # Apply transforms
        if self.use_transforms:
            if self.use_sdf and label is not None:
                image, label, sdf_label = self.transforms(image, label, sdf_label=sdf_label)
            else:
                image, label = self.transforms(image, label)

        # Z-score normalisation
        image = image - image.mean()
        image = image / image.std()

        if self.use_sdf and label is not None:
            label = torch.cat([label, sdf_label], dim=0)

        # If there is no label, create dummy label
        label = label if label is not None else torch.tensor(0)

        if self.inference:
            return image, label, self.meta[idx]

        return image, label
