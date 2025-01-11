import io
import os
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
import skimage
import torch
import yaml
import decimal
import SimpleITK as sitk
from skimage import measure
from tqdm import tqdm

import trimesh

import wandb
from archive.utils.datasets import MRImageDataset


def read_yaml(path):
    with open(path, "r") as stream:
        try:
            args = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise exc

    return args


def nested_set(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


def convert_str(val):
    try:
        val = decimal.Decimal(val)
        if val.as_tuple().exponent == 0:
            return int(val)
        else:
            return float(val)
    except decimal.InvalidOperation:
        return val


def parse_custom_args(args, unk_args):
    if args.config_file:
        args = read_yaml(args.config_file)

    args["sweep_run"] = False

    # Parse sweep args
    if unk_args and unk_args[0] == "--sweep_run":
        args["sweep_run"] = True
        for unk in unk_args[1:]:
            top_level, sub_levels = unk.split("=")
            top_level = top_level.replace("--", "")
            sub_levels = eval(sub_levels)

            for sub_level, sub_value in sub_levels.items():
                args[top_level][sub_level] = sub_value

        return args

    # Command line arguments used to overwrite defaults
    for unk in unk_args:
        unk = unk.replace("--", "")
        arg_name, val = unk.split("=")
        arg_names = arg_name.split(".")

        val = [convert_str(val) for val in val.split(" ")]
        if len(val) == 1 and arg_name not in (
            "training.losses",
            "training.loss_weights",
        ):
            val = val[0]

        nested_set(args, arg_names, val)

    return args


def load_checkpoint(path_to_ckpt):
    api = wandb.Api()
    artifact = api.artifact(path_to_ckpt)
    artifact_dir = artifact.download()
    return artifact_dir + "/model.ckpt"


def fix_ckpt(path, device):
    """Dirty way of removing _orig_mod prefix (introduced by torch.compile())"""
    checkpoint = torch.load(path, map_location=device)
    checkpoint["state_dict"] = {
        k.replace("._orig_mod", ""): v for k, v in checkpoint["state_dict"].items()
    }
    torch.save(checkpoint, path)


def calc_stats(seed_1, seed_2, seed_3):
    import numpy as np

    temp = np.array([*seed_1, *seed_2, *seed_3])
    print(f"{round(temp.mean(), 4)} ± {round(temp.std(), 4)}")


def get_datasets(args, inference=False):
    train_dataset = MRImageDataset(
        img_dir=args["data"]["train"]["img_dir"],
        ann_dir=args["data"]["train"]["ann_dir"],
        use_transforms=True,
        args=args,
        inference=inference,
        use_sdf=args["model"]["dual_head"],
    )
    test_dataset = MRImageDataset(
        img_dir=args["data"]["test"]["img_dir"],
        ann_dir=args["data"]["test"]["ann_dir"],
        use_transforms=False,
        args=args,
        inference=inference,
        use_sdf=args["model"]["dual_head"],
    )

    # Do LOO cross validation
    test_idx = args["training"].get("LOO_fold", None)
    if test_idx is not None:
        # Add all data together
        all_imgs = [*train_dataset.imgs, *test_dataset.imgs]
        all_labels = [*train_dataset.labels, *test_dataset.labels]

        # Pick one test case based on fold
        test_idx = args["training"]["LOO_fold"]
        test_img, test_label = [all_imgs.pop(test_idx)], [all_labels.pop(test_idx)]

        # Update full (train and val) dataset
        train_dataset.imgs = all_imgs
        train_dataset.labels = all_labels

        # Update test dataset
        test_dataset.imgs = test_img
        test_dataset.labels = test_label

    # Split the data
    val_dataset = train_dataset.create_split()
    # train_size = int(0.8 * len(train_dataset))
    # val_size = len(train_dataset) - train_size

    # Use complete dataset for SSL training; no validation needed
    # train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    return train_dataset, val_dataset, test_dataset


def report_results(results, run):
    for split, split_results in results.items():
        print("-" * 40)
        for metric, (metric_mean, metric_std) in split_results.items():
            print(
                f"{metric} on {split}: {round(metric_mean, 4)} ± {round(metric_std, 4)}"
            )
            run.summary[f"{split}/{metric}_mean"] = metric_mean
            run.summary[f"{split}/{metric}_std"] = metric_std
        print("-" * 40)


def set_meta_from_dict(img, meta):
    img.SetDirection(meta["direction"][0].double().numpy())
    img.SetSpacing(meta["spacing"][0].double().numpy())
    img.SetOrigin(meta["origin"][0].double().numpy())
    return img


def set_meta_from_header(img, meta):
    img.SetDirection(meta.GetDirection())
    img.SetSpacing(meta.GetSpacing())
    img.SetOrigin(meta.GetOrigin())
    return img


def save_nii(img_array, meta, path):
    sitk_img = sitk.GetImageFromArray(img_array)

    # Set meta info
    if isinstance(meta, dict):
        sitk_img = set_meta_from_dict(sitk_img, meta)
    elif isinstance(meta, sitk.Image):
        sitk_img = set_meta_from_header(sitk_img, meta)
    else:
        raise NotImplementedError(f"Not implemented to set meta from {type(meta)}")

    return sitk.WriteImage(sitk_img, path)


def torch_from_nii(path):
    sitk_img = sitk.ReadImage(path)
    torch_img = torch.from_numpy(sitk.GetArrayFromImage(sitk_img))
    return torch_img, sitk_img


def skeletonize_data(current_label_dir, new_label_dir):
    os.makedirs(new_label_dir, exist_ok=True)

    for dir, _, files in os.walk(current_label_dir):
        for file in files:
            if file.endswith(".nii"):
                current_label, current_header = torch_from_nii(os.path.join(dir, file))
                new_label = skimage.morphology.skeletonize_3d(current_label)
                save_nii(new_label, current_header, os.path.join(new_label_dir, file))


def extend_dataset():
    dice_metric = Dice(ignore_index=0, average="micro")

    main_dir = "predictions/6t0slmlx/"
    files_to_add = []
    for sub_dir, _, files in os.walk(main_dir):
        if "inference_PT_" not in sub_dir or not sub_dir.endswith("/pred"):
            continue

        # Pick out reference file (frame 80)
        ref = [f for f in files if f.endswith("_080.nii")][0]
        files.remove(ref)
        ref, _ = torch_from_nii(f"{sub_dir}/{ref}")

        # Score all predictions
        dice_scores = dict()
        for file in tqdm(files, desc=sub_dir):
            comp_file, _ = torch_from_nii(f"{sub_dir}/{file}")
            dice_scores[file] = dice_metric(comp_file, ref).item()

        # Sort based on ascending score
        dice_scores = dict(sorted(dice_scores.items(), key=lambda item: item[1]))

        # Add the top-10 lowest scores
        for file in list(dice_scores.keys())[:10]:
            files_to_add.append(f"{sub_dir}/{file}")

    # Copy files to new dataset
    new_data_path = "../../Data/MR_extended_2"
    os.makedirs(f"{new_data_path}/imagesTr/", exist_ok=True)
    os.makedirs(f"{new_data_path}/labelsTr/", exist_ok=True)

    for file in files_to_add:
        prefix = file.split("/")[2].replace("inference_", "")
        main_path, img_name = file.split("pred/")

        shutil.copy(
            f"{main_path}/imgs/{img_name}",
            f"{new_data_path}/imagesTr/{prefix}_{img_name}",
        )
        shutil.copy(
            f"{main_path}/pred/{img_name}",
            f"{new_data_path}/labelsTr/{prefix}_{img_name}",
        )


def plot_marching_cubes(sdf, meta, level=0.0, voxel_size=1.0):
    # Use marching cubes to obtain the surface mesh of these ellipsoids
    verts, faces, normals, values = measure.marching_cubes(
        sdf, level, gradient_direction="ascent"
    )

    min_idx = np.floor(np.min(verts, axis=0)).astype(int)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh_surface = mesh.voxelized(voxel_size).fill().matrix

    sdf_surface = np.zeros_like(sdf)
    max_idx = np.array(mesh_surface.shape) + min_idx

    sdf_surface[
        min_idx[0] : max_idx[0], min_idx[1] : max_idx[1], min_idx[2] : max_idx[2]
    ] = mesh_surface
    sdf_surface = sdf_surface.astype(np.uint8)
    # save_nii(sdf_surface, meta, "../UNet/predictions/z58aezi6_extend_data_real_sdf/test/pred_sdf/sdf_surface.nii")

    return sdf_surface


def set_seed(args):
    torch.manual_seed(args["training"]["seed"])
    torch.cuda.manual_seed(args["training"]["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args["training"]["seed"])
    np.random.seed(args["training"]["seed"])


def cleanup_stored_models(wandb_logger, args):
    api = wandb.Api()
    runs = api.run(
        f"{wandb_logger.experiment.entity}/{wandb_logger.experiment.project_name()}/runs/{wandb_logger.experiment._run_id}"
    )

    # Delete artifacts without aliases (best, latest), or all artifacts when the run was a sweep
    deleted = 0
    for v in runs.logged_artifacts():
        if len(v.aliases) == 0:
            v.delete()
            deleted += 1

    print(f"Deleted {deleted} artifacts")


def main_train_loop(model, args):
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader
    from pytorch_lightning.callbacks import (
        LearningRateMonitor,
        ModelCheckpoint,
        EarlyStopping,
        StochasticWeightAveraging,
    )
    from pytorch_lightning.loggers import WandbLogger

    # Get datasets
    train_dataset, val_dataset, _ = get_datasets(args)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args["training"]["batch_size"],
        shuffle=True,
        persistent_workers=True,
        num_workers=5,
        pin_memory=True,
    )  # num_workers=args['training']['num_workers'])
    val_loader = DataLoader(
        val_dataset,
        batch_size=args["training"]["batch_size"],
        shuffle=False,
        persistent_workers=True,
        num_workers=5,
        pin_memory=True,
    )  # num_workers=args['training']['num_workers'])

    # Callbacks
    monitor = "train/loss" if args["data"]["sanity_check"] else "val/loss"
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor, mode="min", save_last=True
    )  # important: save_last to be turned on, otherwise no checkpoint of last model
    early_stop_callback = EarlyStopping(
        monitor=monitor, mode="min", patience=75, verbose=True
    )
    swa_callback = StochasticWeightAveraging(swa_lrs=model.lr, swa_epoch_start=200)

    wandb_logger = WandbLogger(
        log_model=not args["sweep_run"], project="BowelSegmentation"
    )
    wandb_logger.experiment.config.update(args)  # Log yaml file parameters

    # Calculate number of training batches
    n_batches = len(train_loader) // args["training"]["batch_size"]

    # PL trainer
    trainer = pl.Trainer(
        max_epochs=args["training"]["epochs"],
        callbacks=[
            checkpoint_callback,
            swa_callback,
            lr_monitor,
        ],  # , early_stop_callback], TODO: temp
        logger=wandb_logger,
        precision="16-mixed",
        log_every_n_steps=50 if 50 <= n_batches else n_batches,
        accumulate_grad_batches=args["training"]["accumulate_grad_batches"],
        accelerator="mps" if torch.backends.mps.is_available() else "auto",
    )

    # tuner = pl.tuner.Tuner(trainer)
    #
    # # Run learning rate finder
    # lr_finder = tuner.lr_find(model, train_dataloaders=train_loader, val_dataloaders=val_loader, mode="linear", num_training=800, max_lr=0.1)
    # fig = lr_finder.plot(suggest=True)
    # plt.savefig("lr.png")
    # model.logger.log_image(key="LR finder", images=["lr.png"])

    # Fit the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Make sure everything is uploaded before evaluating, best model might not be uploaded otherwise!
    wandb_logger.experiment.finish()

    # Remove unused models to clean up wandb
    cleanup_stored_models(wandb_logger, args)


# def shortest_path(data, points_path):
#     skel = medial_axis(data)
#
#     pos = {i: val for i, val in enumerate(skel.vertices)}
#
#     g = nx.Graph()
#     g.add_nodes_from(pos.keys())
#     g.add_edges_from(list(map(tuple, skel.edges)))
#
#     orders = []
#     ends, branches = [], []
#     for a in g.nodes():
#         order = len(g.edges(a))
#         if order == 1:
#             ends.append(a)
#         elif order == 3:
#             branches.append(a)
#
#     # Some graph checks
#     n_cycles = len(list(nx.simple_cycles(g)))
#     connected_comps = nx.number_connected_components(g)
#     assert n_cycles == 0, f"Found {n_cycles}! Expect zero cycles. "
#     assert connected_comps == 1, f"To many connected components found: {connected_comps}. Expect 1 connected component"
#
#     import pandas as pd
#     # skels.vertices is z,y,x, and points are in x,y,z
#     points = pd.read_csv(points_path, header=None).to_numpy()
#     points[:, [0, 2]] = points[:, [2, 0]]  # Go from x,y,z to z,y,x
#     points = points * np.array([2, 1, 1])  # Scale z coordinate
#
#     all_paths, total_path = [], set()
#     for i in range(0, points.shape[0]-1, 2):
#         # Multiply final dimension by 2 because of points being in [160,160,50]
#         start = points[i]
#         end = points[i+1]
#
#         # Find closest node
#         start_idx = np.argmin(np.linalg.norm(skel.vertices - start, axis=1))
#         end_idx = np.argmin(np.linalg.norm(skel.vertices - end, axis=1))
#
#         # Find the shortest path
#         best_path = nx.shortest_path(g, source=start_idx, target=end_idx)
#         all_paths.append((start_idx, end_idx, best_path))
#
#         total_path.update(set(best_path))
#
#     # VISUALIZE
#     # Extract node and edge positions from the layout
#     node_xyz = np.array([pos[v] for v in sorted(g)])
#     edge_xyz = np.array([(pos[u], pos[v]) for u, v in g.edges()])
#
#     # Create the 3D figure
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")
#
#     # Plot the nodes - alpha is scaled by "depth" automatically
#     # ax.scatter(*node_xyz.T, s=10, ec="w")
#
#     for n in ends:
#         ax.scatter(*pos[n], s=50, c='red', ec="w")
#
#     for n in branches:
#         ax.scatter(*pos[n], s=50, c='green', ec="w")
#     # for start_idx, end_idx, path in all_paths:
#     #     c = np.random.rand(3,)
#     #     path_xyz = np.array([(pos[u], pos[v]) for u, v in g.subgraph(path).edges()])
#     #
#     #     ax.scatter(*pos[start_idx], s=50, c=c, ec="w")
#     #     ax.scatter(*pos[end_idx], s=50, c=c, ec="w")
#     #
#     #     for vizedge in path_xyz:
#     #         ax.plot(*vizedge.T, color=c)
#
#     # Plot the edges
#     for vizedge in edge_xyz:
#         ax.plot(*vizedge.T, color="tab:gray", alpha=.3)
#
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_zlabel("z")
#
#     fig.tight_layout()
#     plt.show()
#
#     return np.array(skel.vertices)[list(total_path)]


# def draw_graph(data, graph_vertices):
#     centerline = np.zeros_like(data)  # 100, 160, 160
#     for center in graph_vertices:
#         radius = 1
#
#         min = np.round(center - radius).astype(int)
#         max = np.round(center + radius).astype(int)
#         centerline[min[0]:max[0], min[1]:max[1], min[2]:max[2]] = 1
#
#     centerline[~data.bool()] = 0
#     centerline = centerline.astype(np.uint8)
#     return centerline


# def skel_3d_points(data, points_path):
#     vertices = shortest_path(data, points_path=points_path)
#     return draw_graph(data, vertices)


# def skel_3d(data):
#     skels = medial_axis(data)
#     return draw_graph(data, skels.vertices)


if __name__ == "__main__":
    from torchmetrics.classification import Dice

    # Perform skeleton transform on all labels
    # skeletonize_data("../../Data/MR/labelsTr", "../../Data/MR/labelsTr_skel")
    # skeletonize_data("../../Data/MR/labelsTs", "../../Data/MR/labelsTs_skel")

    # Extend dataset with time frame images
    # extend_dataset()

    # sdf, meta = torch_from_nii("../UNet/predictions/z58aezi6_extend_data_real_sdf/test/pred_sdf/pt_002.nii")
    # pred, _ = torch_from_nii("../UNet/predictions/z58aezi6_extend_data_real_sdf/test/pred/pt_002.nii")
    # gt, _ = torch_from_nii("../UNet/predictions/z58aezi6_extend_data_real_sdf/test/label/pt_002.nii")

    # sns.displot(sdf.numpy().flatten())
    # plt.show()

    # current_label, header = torch_from_nii("/Users/thomasvanorden/Documents/UvA Master Artificial Intelligence/Jaar 3/Thesis/Data/annotations_v2/Bowel_06v2.nii")
    current_label, header = torch_from_nii(
        "/Users/thomasvanorden/Documents/UvA Master Artificial Intelligence/Jaar 3/Thesis/Data/CT/Bowel_19v3.nii"
    )
    # current_label, header = torch_from_nii("/Users/thomasvanorden/Documents/UvA Master Artificial Intelligence/Jaar 3/Thesis/Data/CT/Segmentation_MD_CT.nii")
    # new_centerline = skel_3d_points(current_label, points_path="/Users/thomasvanorden/Documents/UvA Master Artificial Intelligence/Jaar 3/Thesis/Data/annotations_v2/start_end.csv")
    # save_nii(new_centerline, header, 'centerline_points.nii')
    exit(1)

    # new_centerline = skel_3d(current_label)
    exit(1)

    new_centerline = skimage.morphology.skeletonize_3d(current_label)
    save_nii(new_centerline, header, "centerline_test_skel.nii")

    # new_surface = plot_marching_cubes(sdf.numpy(), meta, level=.25, voxel_size=1.0)  # 0.25
    # new_surface = torch.from_numpy(new_surface)

    # dice_metric = Dice(ignore_index=0, average='micro')
    # print("DICE(pred, GT)", dice_metric(pred.float(), gt.int()))
    # print("DICE(sdf_surface, GT)", dice_metric(new_surface.float(), gt.int()))
