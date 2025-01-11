import argparse
import shutil
import random

import numpy as np
from torch.utils.data import DataLoader
import wandb
import torch
import os

from tqdm import tqdm

from archive.MAE.pl_wrappers import MAELightningWrapper
from archive.MAE.train import get_model
from archive.utils.utils import get_datasets, report_results, save_nii


def load_model(run_link, device):
    api = wandb.Api()
    run = api.run(run_link)
    run_args = run.config

    # Set seed
    torch.manual_seed(run_args["training"]["seed"])
    if device == "cuda":
        torch.cuda.manual_seed(run_args["training"]["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(run_args["training"]["seed"])
    np.random.seed(run_args["training"]["seed"])

    artifact = api.artifact(
        f"{run.entity}/{run.project}/model-{run.id}:best", type="model"
    )
    artifact_dir = artifact.download()

    model = get_model(run_args)
    model = MAELightningWrapper.load_from_checkpoint(
        f"{artifact_dir}/model.ckpt",
        model=torch.compile(model),
        args=run_args,
        map_location=device,
    )
    return model, run_args, run.id, artifact_dir, run


def predict(
    model,
    run_id,
    split,
    data_loader,
    device="gpu",
    save_results=False,
    report_instance=False,
):
    all_L1 = []

    # Predict
    with torch.no_grad():
        for i, (img, label, meta) in tqdm(
            enumerate(data_loader), desc=split, total=len(data_loader)
        ):
            # Predict
            img = img.to(device)
            L1, pred = model(img)
            L1 = L1.detach().item()

            all_L1.append(L1)

            if report_instance:
                print(f"{split}:{i} L1 score: {round(L1, 4)}")

            # Save img, label, and prediction
            if save_results:
                os.makedirs(f"predictions/{run_id}/{split}/img", exist_ok=True)
                os.makedirs(f"predictions/{run_id}/{split}/label", exist_ok=True)
                os.makedirs(f"predictions/{run_id}/{split}/pred", exist_ok=True)

                save_nii(
                    img[0][0],
                    meta,
                    f"predictions/{run_id}/{split}/img/{meta['name'][0]}.nii",
                )
                save_nii(
                    label[0][0],
                    meta,
                    f"predictions/{run_id}/{split}/label/{meta['name'][0]}.nii",
                )
                save_nii(
                    pred.int()[0][0],
                    meta,
                    f"predictions/{run_id}/{split}/pred/{meta['name'][0]}.nii",
                )

    all_L1 = np.array(all_L1)

    return {"L1": [all_L1.mean(), all_L1.std()]}


def evaluate(run_link, device, save_results=False, report_instance=False):
    # Get best model
    model, args, run_id, model_dir, run = load_model(run_link, device)

    # Get datasets
    train_dataset, val_dataset, test_dataset = get_datasets(args, inference=True)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=1
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=1
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=1
    )

    model = model.eval().to(device)
    train_scores = predict(
        model,
        run_id,
        "train",
        train_loader,
        device=device,
        save_results=save_results,
        report_instance=report_instance,
    )
    val_scores = predict(
        model,
        run_id,
        "val",
        val_loader,
        device=device,
        save_results=save_results,
        report_instance=report_instance,
    )
    test_scores = predict(
        model,
        run_id,
        "test",
        test_loader,
        device=device,
        save_results=save_results,
        report_instance=report_instance,
    )

    results = {"train": train_scores, "val": val_scores, "test": test_scores}

    report_results(results, run)
    run.update()

    # Remove downloaded model
    if "artifacts/model" in model_dir:
        shutil.rmtree(model_dir)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, help="Link to wandb run", required=True)
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save the imgs, labels, and predictions as NIFTI",
    )
    parser.add_argument(
        "--report_instance",
        action="store_true",
        help="Report metrics also per instance",
    )
    args = parser.parse_args()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    evaluate(
        args.run,
        device,
        save_results=args.save_results,
        report_instance=args.report_instance,
    )
