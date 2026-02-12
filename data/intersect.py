import argparse
from datasets import load_from_disk


def intersect(data_paths: list[str]):
    datasets = []
    for path in data_paths:
        ds = load_from_disk(path)
        datasets.append(ds)

    idx_sets = [set(ds["idx"]) for ds in datasets]

    common_indices = set.intersection(*idx_sets)

    print(f"Found {len(common_indices)} common samples across {len(datasets)} datasets")

    inter_datasets = []
    for ds in datasets:
        filtered_ds = ds.filter(lambda x: x["idx"] in common_indices)
        inter_datasets.append(filtered_ds)

    for i, path in enumerate(data_paths):
        _dir = path.strip("/").split("/")[-1]
        subdir = ""
        if args.subdir is not None:
            subdir = args.subdir + "/"
        out_path = f"datasets/intersected/{subdir}{_dir}"
        inter_datasets[i].save_to_disk(out_path)
        print(f"Saved intersected dataset to {out_path}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_paths", type=str, nargs="+")
    parser.add_argument("--subdir", type=str, default=None)

    args = parser.parse_args()

    intersect(args.data_paths)
