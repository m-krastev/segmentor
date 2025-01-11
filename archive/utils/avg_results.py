"""
Assumes folder structure
someFolder/
---folder1/
-----results.json
---folder2/
-----results.json
---folder3/
-----results.json
"""
import argparse
import json
import os
import numpy as np


def read_result(path):
    with open(path, "r") as f:
        results = json.load(f)
    return results


def avg_results(path):
    all_res = []
    for folder in os.listdir(path):
        if not os.path.isdir(f"{path}/{folder}"):
            continue

        result = read_result(f"{path}/{folder}/results.json")
        all_res.append(result)

    avg_res = all_res[0].copy()
    for split, metrics in avg_res.items():
        for metric in metrics.keys():
            all_values = np.array([res[split][metric] for res in all_res], dtype=float)
            avg_res[split][metric] = [round(all_values.mean(), 4), round(all_values.std(), 4)]

    with open(f"{path}/avg_results.json", "w") as f:
        json.dump(avg_res, f, indent=4)
    print(avg_res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_path", required=True, help="Path to model's prediction folder")
    args = parser.parse_args()

    avg_results(args.prediction_path)
