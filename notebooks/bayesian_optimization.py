from bayes_opt import BayesianOptimization
import subprocess
import json
import pandas as pd


def run_graph_approach(idx, thetav, thetad, sigmas, edge_threshold):
    filename_ct = f"../data/bowelseg/s{idx:04d}/ct.nii.gz"
    filename_gt = f"../data/bowelseg/s{idx:04d}/segmentations/small_bowel.nii.gz"
    start_volume = f"../data/bowelseg/s{idx:04d}/segmentations/duodenum.nii.gz"
    end_volume = f"../data/bowelseg/s{idx:04d}/segmentations/colon.nii.gz"
    output_dir = f"results/s{idx:04d}"

    args = [
        "python",
        "graph_approach.py",
        "--sigmas",
        int(sigmas),
        "--thetav",
        thetav,
        "--thetad",
        thetad,
        "--edge_threshold",
        edge_threshold,
        "--precompute",
        "--use_rustworkx",
        "--filename_ct",
        filename_ct,
        "--filename_gt",
        filename_gt,
        "--start_volume",
        start_volume,
        "--end_volume",
        end_volume,
        "--output_dir",
        output_dir,
    ]

    subprocess.run(args, check=True, shell=True, capture_output=True)


def compile_results(thetav, thetad, sigmas, edge_threshold):
    names = [1, 4, 6, 9, 10, 11, 12, 13, 14, 15, 16, 19]
    for idx in names:
        run_graph_approach(idx, thetav, thetad, sigmas, edge_threshold)

    results = []
    for idx in names:
        metrics_file = f"results/s{idx:04d}/metrics.json"
        with open(metrics_file, "r") as f:
            results.append(json.load(f))

    df = pd.DataFrame(results)
    metrics = ["average_gradient", "dice_overlap"]
    # Normalize the metrics using min-max scaling
    for metric in metrics:
        df[metric] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())

    print(df)

    # Define compound score
    score = 4 * df["average_gradient"].mean() + df["dice_overlap"].mean()
    print(
        f"Score: {score}, average_gradient: {df['average_gradient'].mean()}, dice_overlap: {df['dice_overlap'].mean()}"
    )
    return score


# Target parameters
# thetav, thetad, sigmas, edge_threshold, supervoxel_size
# thetav: 2-8
# thetad: 3-10
# sigmas: 1-5
# edge_threshold: 0.1-0.4

pbounds = {
    "thetav": (2, 8),
    "thetad": (3, 10),
    "sigmas": (1, 5),
    "edge_threshold": (0.1, 0.4),
}

optimizer = BayesianOptimization(
    f=compile_results,
    pbounds=pbounds,
    verbose=2,
    random_state=1,
)

optimizer.maximize()

print(optimizer.max)
for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))
