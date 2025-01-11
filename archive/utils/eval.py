"""
Assumes folder structure
someModel/
---Train/
-----seg/
-------pt_001.nii
-----sdf/
-------pt_001.nii
---Val/
-----seg/
-------pt_010.nii
-----sdf/
-------pt_010.nii
---Test/
-----seg/
-------pt_021.nii
-----sdf/
-------pt_021.nii
"""

import argparse
import os
import json

import numpy as np
import torch
from sklearn.metrics import (
    precision_recall_fscore_support,
    precision_recall_curve,
    PrecisionRecallDisplay,
)
from tqdm import tqdm

from archive.UNet.model.pl_wrappers import DiceLoss
from archive.utils.datasets import get_sdf
from archive.utils.utils import torch_from_nii
from torchmetrics.classification import Dice


dice_metric = Dice(ignore_index=0, average="micro")


def sdf2isdf(sdf):
    return torch.abs(sdf - sdf.max())


def score(pred_seg, pred_sdf, gt_seg, gt_sdf=None):
    # Calculate DICE
    dice = dice_metric(pred_seg, gt_seg.int())

    # Calculate Hausdorff proxy
    pred_seg = (pred_seg >= 0.5).int()  # convert probabilities to binary
    sdf_temp = get_sdf(pred_seg[None, ...])[0]
    gt_sdf = get_sdf(gt_seg[None, ...])[0]
    pred2gt = sdf_temp[gt_seg.bool()].max()
    gt2pred = gt_sdf[pred_seg.bool()].max()
    hausdorff = max(pred2gt, gt2pred)

    precision, recall, _, _ = precision_recall_fscore_support(
        gt_seg.bool().numpy().flatten(),
        pred_seg.bool().numpy().flatten(),
        average="binary",
    )

    # Calculate L1 & MSE
    l1, mse = None, None
    if pred_sdf is not None:
        l1 = torch.mean(torch.abs(gt_sdf - pred_sdf))
        mse = torch.mean((sdf2isdf(gt_sdf) - sdf2isdf(pred_sdf)) ** 2)

    return dice, hausdorff, precision, recall, l1, mse


def eval_set(pred_path, seg_path, sdf_path=None):
    scores = {
        "VolDist": [],
        "DICE": [],
        "Precision": [],
        "Recall": [],
        "L1": [],
        "MSE": [],
    }

    for pt in tqdm(
        os.listdir(f"{pred_path}/seg"), desc=f"Evaluating {pred_path.split('/')[-1]}"
    ):
        if not pt.endswith(".nii"):
            continue

        pred_seg, _ = torch_from_nii(f"{pred_path}/seg/{pt}")
        if os.path.isdir(f"{pred_path}/sdf"):
            pred_sdf, _ = torch_from_nii(f"{pred_path}/sdf/{pt}")
        else:
            pred_sdf = None

        gt_seg, _ = torch_from_nii(f"{seg_path}/{pt}")
        gt_sdf = None
        if sdf_path is not None:
            gt_sdf, _ = torch_from_nii(f"{sdf_path}/{pt}")

        dice, hausdorff, precision, recall, l1, mse = score(
            pred_seg, pred_sdf, gt_seg, gt_sdf
        )

        # Update scores
        scores["DICE"].append(dice)
        scores["VolDist"].append(hausdorff)
        scores["Precision"].append(precision)
        scores["Recall"].append(recall)
        if l1:
            scores["L1"].append(l1)
        if mse:
            scores["MSE"].append(mse)

    for k, v in scores.items():
        scores[k] = np.around(np.array(v, dtype=float).mean(), 3) if v else np.nan

    return scores


def eval_model(preds_path, seg_path, sdf_path=None):
    model_scores = dict()

    model_scores["train"] = eval_set(f"{preds_path}/Train", seg_path, sdf_path)
    model_scores["val"] = eval_set(f"{preds_path}/Val", seg_path, sdf_path)
    model_scores["test"] = eval_set(f"{preds_path}/Test", seg_path, sdf_path)

    return model_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction_path", required=True, help="Path to model's prediction folder"
    )
    parser.add_argument(
        "--gt_seg_path",
        default="../../Results/GTs/seg/",
        help="Path to segmentation ground truth folder",
    )
    parser.add_argument(
        "--gt_sdf_path", required=False, help="Path to sdf ground truth folder"
    )

    args = parser.parse_args()

    results = eval_model(args.prediction_path, args.gt_seg_path, args.gt_sdf_path)
    print(results)

    with open(f"{args.prediction_path}/results.json", "w") as f:
        json.dump(results, f, indent=4)
