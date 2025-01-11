import torch
import torchvision
from torchmetrics import Dice

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
import pytorch_lightning as pl
from scipy.ndimage.morphology import distance_transform_edt as edt
import numpy as np

from archive.UNet.model.unified_focal_loss import AsymmetricUnifiedFocalLoss


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "none",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

        Args:
            inputs (Tensor): A float tensor of arbitrary shape, sigmoid already applied so in [0, 1].
                    The predictions for each example.
            targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha (float): Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default: ``0.25``.
            gamma (float): Exponent of the modulating factor (1 - p_t) to
                    balance easy vs hard examples. Default: ``2``.
            reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                    ``'none'``: No reduction will be applied to the output.
                    ``'mean'``: The output will be averaged.
                    ``'sum'``: The output will be summed. Default: ``'none'``.
        Returns:
            Loss tensor with the reduction option applied.
        """
        # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # Check reduction option and return loss accordingly
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss


class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(img.shape[0]):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)

        Assumes pred to be [0,1] (sigmoid)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        pred_dt = torch.from_numpy(self.distance_field(pred.detach().cpu())).float()
        target_dt = torch.from_numpy(self.distance_field(target.detach().cpu())).float()

        pred_error = (pred - target) ** 2
        pred_error = pred_error.detach().cpu()

        distance = pred_dt**self.alpha + target_dt**self.alpha

        dt_field = pred_error * distance
        loss = dt_field.mean()

        return loss


class DiceLoss(nn.Module):
    def __init__(self, background_index=None):
        super().__init__()
        self.background_index = background_index

    def forward(self, inputs, targets, smooth=1):
        """Assumes inputs are already [0,1] (sigmoid)"""

        # flatten label and prediction tensors
        inputs = inputs.flatten()
        targets = targets.flatten()

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


class UnetLightningWrapper(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model

        # Add to logging
        self.hparams.model_name = model.__class__.__name__

        self.dice_metric = Dice(ignore_index=0, average="micro")

        self.use_sdf = args["model"]["dual_head"]
        self.args = args
        self.lr = args["optimizer"]["lr"]

        # Losses
        self.dice_loss = DiceLoss(background_index=0)
        self.hausdorff_loss = HausdorffDTLoss()
        self.mse_loss = nn.MSELoss(reduction="none")
        self.L1_loss = nn.L1Loss(reduction="none")

        self.__device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )  # TODO: should be handled better

    def forward(self, img):
        outputs = self.model(img)
        return outputs

    def common_step(self, batch, is_testing=False):
        logits = self.forward(batch[0])
        labels = batch[1]

        # Model outputs raw values, no logits
        seg_logits = logits[:, 0].sigmoid()
        if self.use_sdf:
            sdf_logits = logits[:, 1]
        else:
            sdf_logits = torch.zeros_like(seg_logits)

        # Calculate DICE and Hausdorff loss
        L_dice = self.dice_loss(seg_logits, labels[:, 0].int())
        L_Hausdorff = self.hausdorff_loss(
            seg_logits[:, None, ...], labels[:, 0][:, None, ...].int()
        )  # Might disable for performance reasons
        per_loss = {"DICE": L_dice.detach(), "Hausdorff": L_Hausdorff.detach()}

        # Calculate dice score
        dice_score = self.dice_metric(seg_logits, labels[:, 0].int())

        if self.use_sdf:
            # Calculate MSE loss
            L_MSE = self.mse_loss(sdf_logits, labels[:, 1]).mean()
            L_1 = self.L1_loss(sdf_logits, labels[:, 1]).mean()

            # L_eikonal = (torch.norm(L_MSE) - 1) ** 2 TODO: is this correct? should it be step-differentiated?

            per_loss["MSE"] = L_MSE.detach()
            per_loss["L1"] = L_1.detach()

            total_loss = (
                self.args["training"]["DICE_weight"] * L_dice
                + self.args["training"]["L1_weight"] * L_1
            )
        else:
            total_loss = L_dice

        # Used for testing
        if is_testing:
            return logits, dice_score, per_loss

        return total_loss, dice_score, per_loss, seg_logits, sdf_logits

    def training_step(self, batch, batch_idx):
        loss, dice_score, per_loss, seg_logs, sdf_logs = self.common_step(batch)
        if batch_idx == 0 and self.args["data"]["sanity_check"]:
            self.logger.log_image(
                key="Training, slice 50",
                images=[
                    batch[0][0][0][50],
                    seg_logs[0][50],
                    batch[1][0][0][50],
                    sdf_logs[0][50],
                ],
                caption=["Image", "SEG prediction", "SEG GT", "SDF prediction"],
            )

        self.log("train/loss", loss)
        self.log("train/dice_score", dice_score)

        for loss_name, loss_val in per_loss.items():
            self.log(f"train/{loss_name}_loss", loss_val)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, dice_score, per_loss, seg_logs, sdf_logs = self.common_step(batch)
        if batch_idx == 0 and not self.args["data"]["sanity_check"]:
            self.logger.log_image(
                key="Validation, slice 50",
                images=[
                    batch[0][0][0][50],
                    seg_logs[0][50],
                    batch[1][0][0][50],
                    sdf_logs[0][50],
                ],
                caption=["Image", "SEG prediction", "SEG GT", "SDF prediction"],
            )

        self.log("val/loss", loss, on_epoch=True)
        self.log("val/dice_score", dice_score, on_epoch=True)

        for loss_name, loss_val in per_loss.items():
            self.log(f"val/{loss_name}_loss", loss_val, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        logits, dice_score, per_loss, _, _ = self.common_step(batch, is_testing=True)
        self.log("test/dice_score", dice_score, on_epoch=True)

        for loss_name, loss_val in per_loss.items():
            self.log(f"test/{loss_name}_loss", loss_val, on_epoch=True)

        return dice_score

    def configure_optimizers(self):
        if self.args["optimizer"]["name"] == "SGD":
            self.optimizer = SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=1e-6,
                nesterov=True,
            )
        elif self.args["optimizer"]["name"] == "Adam":
            self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        elif self.args["optimizer"]["name"] == "AdamW":
            self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError(
                f"Optimizer {self.args['optimizer']['name']} not implementend"
            )

        if self.args["data"]["sanity_check"]:
            self.args["lr_scheduler"]["patience"] = self.args["training"]["epochs"] // 4

        self.scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.optimizer,
            step_size_up=50,
            base_lr=self.lr * 0.5,
            max_lr=self.lr * 1.5,
            cycle_momentum=False,
        )
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9, verbose=True)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',
        #                                                             patience=self.args['lr_scheduler']['patience'],
        #                                                             verbose=True)

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
            "interval": "step",
            "monitor": "val/loss",
        }
