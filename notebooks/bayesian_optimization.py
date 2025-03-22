from bayes_opt import BayesianOptimization
import subprocess
import json
import pandas as pd


def run_graph_approach(idx, thetav, thetad, sigmas, edge_threshold = 0.1, supervoxel_size=216):
    filename_ct = f"../data/bowelseg/s{idx:04d}/ct.nii.gz"
    filename_gt = f"../data/bowelseg/s{idx:04d}/segmentations/small_bowel.nii.gz"
    start_volume = f"../data/bowelseg/s{idx:04d}/segmentations/duodenum.nii.gz"
    end_volume = f"../data/bowelseg/s{idx:04d}/segmentations/colon.nii.gz"
    output_dir = f"results/s{idx:04d}"

    args = [
        "python",
        "graph_approach.py",
        "--sigmas",
        round(sigmas),
        "--thetav",
        round(thetav),
        "--thetad",
        round(thetad),
        "--edge_threshold",
        edge_threshold,
        "--delta", 5000,
        "--supervoxel_size", int(supervoxel_size),
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
        "--output",
        output_dir,
    ]

    args = list(map(str, args))
    result = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        executable="/home/mkrastev1/segmentor/.venv/bin/python",
        text=True,
    )


def compile_results(thetav, thetad, sigmas, edge_threshold = 0.1, supervoxel_size=216):
    names = [1, 4, 6, 9, 10, 11, 12, 13]
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
    # for metric in metrics:
    #     df[metric] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())

    print(df)

    # Define compound score
    score = (- df["average_gradient"].mean())
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
    "thetav": (3, 8, int),
    "thetad": (4, 10, int),
    "sigmas": (1, 5, int),
    "supervoxel_size": (100, 500, int),
}

optimizer = BayesianOptimization(
    f=compile_results,
    pbounds=pbounds,
    verbose=2,
    random_state=1,
)

optimizer.maximize(n_iter=40)

for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))
print(optimizer.max)

compile_results(**optimizer.max["params"])

## Optimal result when trying to minimize the average_gradient and maximize the dice_overlap
# {
#     "target": np.float64(-0.10015069658870435),
#     "params": {"thetav": np.float64(3.0), "thetad": np.float64(5.0), "sigmas": np.float64(5.0)},
# }
# compile_results(thetav=3, thetad=5, sigmas=5)