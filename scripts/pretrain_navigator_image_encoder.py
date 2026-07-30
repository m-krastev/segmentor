#!/usr/bin/env python3
"""Pretrain Navigator's 3-D encoder from unlabeled CT/filter patches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from navigator.models.memory import NavigatorDenoisingAutoencoder


CHANNEL_NAMES = (
    "ct_clipped",
    "dark_tubularity",
    "bright_tubularity",
    "band_pass",
    "gradient",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-case-id", action="append", required=True)
    parser.add_argument("--validation-case-id", action="append", required=True)
    parser.add_argument("--patch-size-vox", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batches-per-case", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mask-fraction", type=float, default=0.4)
    parser.add_argument("--mask-cube-size", type=int, default=4)
    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp-dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensorboard-log-dir", type=Path)
    return parser.parse_args()


def load_image_channels(data_dir: Path, case_id: str) -> np.ndarray:
    """Load five image-only channels; no anatomical label path is opened."""

    case_dir = data_dir / case_id
    ct = nib.load(case_dir / "ct.nii.gz").get_fdata(dtype=np.float32)
    filters = nib.load(
        case_dir / "cache" / "navigation_filters-v1-mm-3-6-9.nii"
    ).get_fdata(dtype=np.float32)
    if filters.shape != (*ct.shape, 4):
        raise ValueError(
            f"{case_id}: filter shape {filters.shape} does not match CT {ct.shape}"
        )
    clipped_ct = (np.clip(ct, -120.0, 180.0) + 120.0) / 300.0
    return np.concatenate(
        [clipped_ct[..., None], np.clip(filters, 0.0, 1.0)],
        axis=-1,
    ).astype(np.float32, copy=False)


def sample_patches(
    channels: np.ndarray,
    *,
    patch_size: int,
    batch_size: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Sample body-centered patches without segmentation or endpoint labels."""

    spatial_shape = np.asarray(channels.shape[:3], dtype=int)
    half = patch_size // 2
    lower = np.full(3, half, dtype=int)
    upper = spatial_shape - (patch_size - half)
    if np.any(upper <= lower):
        raise ValueError(
            f"Patch size {patch_size} does not fit volume {tuple(spatial_shape)}"
        )

    patches = []
    attempts = 0
    maximum_attempts = batch_size * 100
    while len(patches) < batch_size and attempts < maximum_attempts:
        center = rng.integers(lower, upper + 1)
        attempts += 1
        # Air clips to exactly zero. Restricting the center to non-air removes
        # trivial all-background reconstruction without using an organ mask.
        if channels[tuple(center)][0] <= 0.01:
            continue
        start = center - half
        stop = start + patch_size
        patch = channels[
            start[0] : stop[0],
            start[1] : stop[1],
            start[2] : stop[2],
        ]
        if np.mean(patch[..., 0] > 0.01) < 0.25:
            continue
        patches.append(np.moveaxis(patch, -1, 0))
    if len(patches) != batch_size:
        raise RuntimeError(
            f"Could sample only {len(patches)}/{batch_size} non-air patches"
        )
    return torch.from_numpy(np.stack(patches))


def random_flip(
    patches: torch.Tensor,
    rng: np.random.Generator,
) -> torch.Tensor:
    for spatial_axis in range(2, 5):
        if rng.random() < 0.5:
            patches = torch.flip(patches, dims=(spatial_axis,))
    return patches


def corrupt_patches(
    target: torch.Tensor,
    *,
    mask_fraction: float,
    mask_cube_size: int,
    noise_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    coarse_shape = tuple(
        max(1, int(np.ceil(size / mask_cube_size)))
        for size in target.shape[-3:]
    )
    coarse_mask = (
        torch.rand(
            target.shape[0],
            1,
            *coarse_shape,
            device=target.device,
        )
        < mask_fraction
    )
    mask = F.interpolate(
        coarse_mask.float(),
        size=target.shape[-3:],
        mode="nearest",
    ).bool()
    channel_means = target.mean(dim=(-3, -2, -1), keepdim=True)
    corrupted = torch.where(mask, channel_means, target)
    if noise_std:
        corrupted = corrupted + noise_std * torch.randn_like(corrupted)
    return corrupted.clamp(0.0, 1.0), mask


def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    voxel_loss = F.smooth_l1_loss(
        prediction,
        target,
        reduction="none",
    )
    mask = mask.to(dtype=voxel_loss.dtype)
    masked_voxel_count = mask.sum(dim=(0, 2, 3, 4)).clamp_min(1.0)
    channel_loss = (voxel_loss * mask).sum(dim=(0, 2, 3, 4))
    channel_loss = channel_loss / masked_voxel_count
    channel_weights = prediction.new_tensor((1.0, 0.5, 0.5, 0.5, 0.5))
    loss = torch.sum(channel_loss * channel_weights) / channel_weights.sum()
    return loss, channel_loss


def make_fixed_validation_batch(
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> torch.Tensor:
    per_case = max(1, args.batch_size // len(args.validation_case_id))
    batches = []
    for case_id in args.validation_case_id:
        channels = load_image_channels(args.data_dir, case_id)
        batches.append(
            sample_patches(
                channels,
                patch_size=args.patch_size_vox,
                batch_size=per_case,
                rng=rng,
            )
        )
    return torch.cat(batches, dim=0)[: args.batch_size]


def make_checkpoint(
    args: argparse.Namespace,
    *,
    best_step: int,
    best_validation_loss: float,
    validation_baseline_loss: float,
    spatial_encoder_state_dict: dict[str, torch.Tensor],
) -> dict:
    return {
        "spatial_encoder_state_dict": spatial_encoder_state_dict,
        "input_channels": len(CHANNEL_NAMES),
        "channel_names": CHANNEL_NAMES,
        "best_step": best_step,
        "best_validation_loss": best_validation_loss,
        "validation_masked_mean_fill_baseline_loss": validation_baseline_loss,
        "train_case_ids": args.train_case_id,
        "validation_case_ids": args.validation_case_id,
        "patch_size_vox": args.patch_size_vox,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "mask_fraction": args.mask_fraction,
        "mask_cube_size": args.mask_cube_size,
        "noise_std": args.noise_std,
        "seed": args.seed,
        "amp_dtype": args.amp_dtype,
        "label_files_read": False,
    }


def save_checkpoint(checkpoint: dict, output: Path) -> None:
    """Atomically replace the best checkpoint after each validation."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(f"{output.suffix}.tmp")
    torch.save(checkpoint, temporary_output)
    temporary_output.replace(output)


def main() -> None:
    args = parse_args()
    requested_cases = args.train_case_id + args.validation_case_id
    if len(set(requested_cases)) != len(requested_cases):
        raise ValueError("Train and validation case IDs must be disjoint and unique")
    if args.patch_size_vox < 16 or args.patch_size_vox % 4:
        raise ValueError("patch_size_vox must be a multiple of four and at least 16")
    if min(
        args.batch_size,
        args.steps,
        args.batches_per_case,
        args.learning_rate,
        args.validation_interval,
        args.mask_cube_size,
    ) <= 0:
        raise ValueError("Counts, learning rate, and mask cube must be positive")
    if not 0 < args.mask_fraction < 1:
        raise ValueError("mask_fraction must be in (0, 1)")
    if args.noise_std < 0 or args.weight_decay < 0:
        raise ValueError("noise_std and weight_decay must be non-negative")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

    model = NavigatorDenoisingAutoencoder(input_channels=len(CHANNEL_NAMES)).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    amp_enabled = device.type == "cuda" and args.amp_dtype != "fp32"
    amp_dtype = (
        torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    )
    scaler = torch.GradScaler(
        enabled=amp_enabled and amp_dtype == torch.float16
    )
    writer = (
        SummaryWriter(str(args.tensorboard_log_dir))
        if args.tensorboard_log_dir
        else None
    )

    validation_rng = np.random.default_rng(args.seed + 1)
    fixed_validation = make_fixed_validation_batch(
        args,
        validation_rng,
    ).to(device)
    fixed_corrupted, fixed_mask = corrupt_patches(
        fixed_validation,
        mask_fraction=args.mask_fraction,
        mask_cube_size=args.mask_cube_size,
        noise_std=args.noise_std,
    )
    with torch.no_grad():
        validation_baseline, _ = reconstruction_loss(
            fixed_corrupted,
            fixed_validation,
            fixed_mask,
        )

    best_validation_loss = float("inf")
    best_step = 0
    best_spatial_state = None
    started = time.perf_counter()
    current_case = None
    current_channels = None
    final_train_loss = float("nan")
    final_channel_loss = None

    try:
        for step in range(1, args.steps + 1):
            block = (step - 1) // args.batches_per_case
            case_id = args.train_case_id[block % len(args.train_case_id)]
            if case_id != current_case:
                current_channels = load_image_channels(args.data_dir, case_id)
                current_case = case_id

            target = sample_patches(
                current_channels,
                patch_size=args.patch_size_vox,
                batch_size=args.batch_size,
                rng=rng,
            )
            target = random_flip(target, rng).to(device)
            corrupted, corruption_mask = corrupt_patches(
                target,
                mask_fraction=args.mask_fraction,
                mask_cube_size=args.mask_cube_size,
                noise_std=args.noise_std,
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                prediction = model(corrupted)
                loss, channel_loss = reconstruction_loss(
                    prediction,
                    target,
                    corruption_mask,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            final_train_loss = float(loss.detach())
            final_channel_loss = channel_loss.detach().float().cpu()

            if writer is not None:
                writer.add_scalar("pretrain/train_loss", final_train_loss, step)
            if step % args.validation_interval == 0 or step == args.steps:
                model.eval()
                with torch.no_grad(), torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    validation_prediction = model(fixed_corrupted)
                    validation_loss, validation_channels = reconstruction_loss(
                        validation_prediction,
                        fixed_validation,
                        fixed_mask,
                    )
                model.train()
                validation_value = float(validation_loss)
                if validation_value < best_validation_loss:
                    best_validation_loss = validation_value
                    best_step = step
                    best_spatial_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.encoder.spatial_state_dict().items()
                    }
                    save_checkpoint(
                        make_checkpoint(
                            args,
                            best_step=best_step,
                            best_validation_loss=best_validation_loss,
                            validation_baseline_loss=float(validation_baseline),
                            spatial_encoder_state_dict=best_spatial_state,
                        ),
                        args.output,
                    )
                if writer is not None:
                    writer.add_scalar(
                        "pretrain/validation_loss",
                        validation_value,
                        step,
                    )
                    for name, value in zip(
                        CHANNEL_NAMES,
                        validation_channels.detach().float().cpu(),
                    ):
                        writer.add_scalar(
                            f"pretrain/validation_{name}",
                            float(value),
                            step,
                        )
                    writer.flush()
                elapsed = time.perf_counter() - started
                print(
                    f"step={step} train={final_train_loss:.6f} "
                    f"validation={validation_value:.6f} "
                    f"baseline={float(validation_baseline):.6f} "
                    f"steps_per_second={step / max(elapsed, 1e-6):.2f}",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()

    if best_spatial_state is None:
        raise RuntimeError("Pretraining produced no validation checkpoint")
    checkpoint = make_checkpoint(
        args,
        best_step=best_step,
        best_validation_loss=best_validation_loss,
        validation_baseline_loss=float(validation_baseline),
        spatial_encoder_state_dict=best_spatial_state,
    )
    save_checkpoint(checkpoint, args.output)
    elapsed = time.perf_counter() - started
    payload = {
        key: value
        for key, value in checkpoint.items()
        if key != "spatial_encoder_state_dict"
    }
    payload.update(
        {
            "output": str(args.output),
            "steps": args.steps,
            "final_train_loss": final_train_loss,
            "final_train_channel_loss": {
                name: float(value)
                for name, value in zip(CHANNEL_NAMES, final_channel_loss)
            },
            "elapsed_seconds": elapsed,
            "steps_per_second": args.steps / elapsed,
            "peak_cuda_allocated_mib": (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda"
                else 0.0
            ),
            "peak_cuda_reserved_mib": (
                torch.cuda.max_memory_reserved(device) / 1024**2
                if device.type == "cuda"
                else 0.0
            ),
        }
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
