import argparse
import json
import shutil
import random
import numpy as np
import wandb
import torch
import os

from torchmetrics import Dice
from tqdm import tqdm
from torch.utils.data import DataLoader

from archive.ViT.train import load_vit_model
from archive.utils.datasets import MRImageDataset
from archive.utils.eval import score
from archive.utils.utils import get_datasets, save_nii, fix_ckpt, set_seed
from archive.UNet.train import load_unet_model


def load_model(run_link, device, model_selection="latest"):
    api = wandb.Api()
    run = api.run(run_link)
    run_args = run.config

    set_seed(run_args)

    artifact = api.artifact(
        f"{run.entity}/{run.project}/model-{run.id}:{model_selection}", type="model"
    )
    artifact_dir = artifact.download()
    print(f"Loading {artifact.metadata['original_filename']}")

    fix_ckpt(f"{artifact_dir}/model.ckpt", device=device)

    if "SwinUNETR" in run_args["model"]["name"]:
        model_cls, model = load_vit_model(run_args, return_class=True)
    elif "UNet3D" in run_args["model"]["name"]:
        model_cls, model = load_unet_model(run_args, return_class=True)
    else:
        raise NotImplementedError(
            f"Loading model {run_args['model']['name']} not implemented"
        )

    model = model_cls.load_from_checkpoint(
        f"{artifact_dir}/model.ckpt",
        model=model,
        args=run_args,
        map_location=device,
        strict=False,
    )
    model = model.eval()

    return model, run_args, run.id, artifact_dir, run


def predict(model, split, data_loader, save_path, device="gpu"):
    scores = {
        "VolDist": [],
        "DICE": [],
        "Precision": [],
        "Recall": [],
        "L1": [],
        "MSE": [],
    }

    if save_path:
        os.makedirs(f"{save_path}/{split}/seg", exist_ok=True)
        os.makedirs(f"{save_path}/{split}/sdf", exist_ok=True)
        print(f"Saving to {save_path}/{split}/seg")

    with torch.no_grad():
        for i, (img, label, meta) in tqdm(
            enumerate(data_loader), desc=split, total=len(data_loader)
        ):
            pt = meta["name"][0]

            # Predict
            img = img.to(device)
            pred = model(img).cpu()  # [1, 2, 100, 160, 160]

            # Check if model outputs both segmentation and sdf prediction
            seg_pred, sdf_pred = pred[0, 0], None
            if pred.shape[1] == 2:
                sdf_pred = pred[0, 1]
            seg_pred = seg_pred.sigmoid()  # convert to [0,1]

            # Score predictions
            sdf_gt = None
            if label.shape[1] == 2:  # [1, 1, 100, 160, 160] or [1, 2, 100, 160, 160]
                sdf_gt = label[0, 1]
            seg_gt = label[0, 0]

            dice, hausdorff, precision, recall, l1, mse = score(
                seg_pred, sdf_pred, seg_gt, sdf_gt
            )

            # Save predictions
            if save_path:
                save_nii(seg_pred.numpy(), meta, f"{save_path}/{split}/seg/{pt}.nii")
                save_nii(sdf_pred.numpy(), meta, f"{save_path}/{split}/sdf/{pt}.nii")

            # Update scores
            scores["DICE"].append(dice)
            scores["VolDist"].append(hausdorff)
            scores["Precision"].append(precision)
            scores["Recall"].append(recall)
            if l1:
                scores["L1"].append(l1)
            if mse:
                scores["MSE"].append(mse)

    # Round scores
    for k, v in scores.items():
        scores[k] = np.array(v, dtype=float).mean() if v else np.nan

    return scores


def select_device(device):
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        if torch.backends.mps.is_available():
            device = "mps"
    else:
        device = "cpu"

    return device


def inference(
    run_link, device, save_path, model_selection="latest", img_path=None, ann_path=None
):
    device = select_device(device)

    # Get best model
    model, args, run_id, model_dir, run = load_model(
        run_link, device, model_selection=model_selection
    )
    model = model.eval().to(device)

    # Get datasets
    if img_path is not None:
        test_dataset = MRImageDataset(
            img_dir=img_path,
            ann_dir=ann_path,
            use_transforms=True,
            args=args,
            inference=True,
            use_sdf=args["model"]["dual_head"],
        )
    else:
        train_dataset, val_dataset, test_dataset = get_datasets(args, inference=True)
        train_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=1
        )
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=1
        )

    # Create dataloaders
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=1
    )

    model_scores = dict()
    model_scores["test"] = predict(model, "Test", test_loader, save_path, device=device)

    if img_path is None:
        model_scores["train"] = predict(
            model, "Train", train_loader, save_path, device=device
        )
        model_scores["val"] = predict(
            model, "Val", val_loader, save_path, device=device
        )

    # Remove downloaded model
    if "artifacts/model" in model_dir:
        shutil.rmtree(model_dir)

    return model_scores


def get_sum_scores(scores):
    # Val might not exist
    val = (
        scores.get("val", dict()).get("DICE", 0)
        + scores.get("val", dict()).get("Precision", 0)
        + scores.get("val", dict()).get("Recall", 0)
    )
    test = (
        scores["test"]["DICE"] + scores["test"]["Precision"] + scores["test"]["Recall"]
    )
    return val + test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, help="Link to wandb run", required=True)
    parser.add_argument(
        "--results_save_path",
        type=str,
        required=True,
        help="Path to save json with evaluation results",
    )
    parser.add_argument(
        "--pred_save_path", type=str, help="Path to save model's predictions"
    )
    parser.add_argument("--img_path", type=str, help="Path to custom data img folder")
    parser.add_argument(
        "--ann_path", type=str, help="Path to custom data annotations folder"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Select device to run inference on"
    )

    args = parser.parse_args()

    results_latest = inference(
        args.run,
        args.device,
        args.pred_save_path,
        model_selection="latest",
        img_path=args.img_path,
        ann_path=args.ann_path,
    )
    results_best = inference(
        args.run,
        args.device,
        args.pred_save_path,
        model_selection="best",
        img_path=args.img_path,
        ann_path=args.ann_path,
    )

    # Determine which model to use
    if get_sum_scores(results_best) >= get_sum_scores(results_latest):
        print("Using model:best")
        results = results_best
    else:
        print("Using model:latest")
        results = results_latest

    # Save results
    os.makedirs(args.results_save_path, exist_ok=True)
    with open(f"{args.results_save_path}/results.json", "w") as f:
        json.dump(results, f, indent=4)
