from archive.UNet.model.pl_wrappers import HausdorffDTLoss, Dice
from archive.utils.datasets import get_sdf
from archive.utils.utils import torch_from_nii
import torch


if __name__ == "__main__":
    gt, _ = torch_from_nii(
        "/Users/thomasvanorden/Documents/UvA Master Artificial Intelligence/Jaar 3/Thesis/Data/Centerlines/labelsTr/pt_020.nii"
    )
    raw_pred, _ = torch_from_nii(
        "/Users/thomasvanorden/Documents/UvA Master Artificial Intelligence/Jaar 3/Thesis/BowelSegmentation/BowelGraph/raw_graph_gt_pt_020.nii"
    )
    pred, header = torch_from_nii(
        "/Users/thomasvanorden/Documents/UvA Master Artificial Intelligence/Jaar 3/Thesis/BowelSegmentation/BowelGraph/020_post_processed_v2.nii"
    )

    hausdorff = HausdorffDTLoss()
    dice = Dice(ignore_index=0, average="micro")

    # Add batch and 1 dim
    gt = gt[None, None, ...]
    raw_pred = raw_pred[None, None, ...]
    pred = pred[None, None, ...]

    sdf_gt = get_sdf(gt[0])[0] * 1.25

    import seaborn as sns
    import matplotlib.pyplot as plt

    sdf_raw = sdf_gt[raw_pred.bool()[0][0]].flatten()
    sdf_pred = sdf_gt[pred.bool()[0][0]].flatten()

    print(sdf_raw.mean(), sdf_raw.sum())
    print(sdf_pred.mean(), sdf_pred.sum())

    sns.distplot(sdf_raw, label="Original")
    sns.distplot(sdf_pred, label="Post-processed")
    plt.legend()
    plt.show()

    print(hausdorff(raw_pred, gt), hausdorff(pred, gt))
    print(dice(raw_pred, gt), dice(pred, gt))

    # pred: (b, 1, x, y, z) or (b, 1, x, y)
    # target: (b, 1, x, y, z) or (b, 1, x, y)
