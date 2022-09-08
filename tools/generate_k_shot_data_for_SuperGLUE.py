"""This script samples K examples randomly without replacement from the original data."""

import argparse
import os
import numpy as np
import jsonlines


def load_datasets(data_dir, tasks):
    datasets = {}
    for task in tasks:
        # SuperGLUE style (jsonl)
        dataset = {}
        dirname = os.path.join(data_dir, task)
        splits = ["train", "val"]
        for split in splits:
            if task in ["MultiRC", "ReCoRD", "WSC"]:
                filename = os.path.join(dirname, f"processed_{split}.jsonl")
            else:
                filename = os.path.join(dirname, f"{split}.jsonl")
            with open(filename, "r+", encoding="utf8") as f:
                lines = list(jsonlines.Reader(f))
            dataset[split] = lines
        datasets[task] = dataset
    return datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--k", type=int, default=16, help="Training examples for each class."
    )
    parser.add_argument(
        "--task",
        type=str,
        nargs="+",
        default=[
            "AX-b",
            "AX-g",
            "BoolQ",
            "CB",
            "COPA",
            "MultiRC",
            "ReCoRD",
            "WiC",
            "WSC",
        ],
        help="Task names",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=[100, 13, 21, 42, 87],
        help="Random seeds",
    )

    parser.add_argument(
        "--data_dir", type=str, default="data/SuperGLUE", help="Path to original data"
    )
    parser.add_argument("--output_dir", type=str, default="data", help="Output path")
    parser.add_argument(
        "--mode",
        type=str,
        default="k-shot",
        choices=["k-shot", "k-shot-10x"],
        help="k-shot or k-shot-10x (10x val set)",
    )

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.mode)

    k = args.k
    print("K =", k)
    datasets = load_datasets(args.data_dir, args.task)

    for seed in args.seed:
        print("Seed = %d" % (seed))
        for task, dataset in datasets.items():
            # Set random seed
            np.random.seed(seed)

            # Shuffle the training set
            print("| Task = %s" % (task))
            train_lines = dataset["train"]
            np.random.shuffle(train_lines)

            # Set up dir
            task_dir = os.path.join(args.output_dir, task)
            setting_dir = os.path.join(task_dir, f"{k}-{seed}")
            os.makedirs(setting_dir, exist_ok=True)

            # Write test splits,
            # SuperGLUE style
            # Use the original valelopment set as the test set (the original test sets are not publicly available)
            for split, lines in dataset.items():
                if split.startswith("train"):
                    continue
                split = split.replace("val", "test")
                with jsonlines.open(
                    os.path.join(setting_dir, f"{split}.jsonl"), "w"
                ) as f:
                    for line in lines:
                        f.write(line)

            if task == "ReCoRD":
                with jsonlines.open(os.path.join(setting_dir, "train.jsonl"), "w") as f:
                    for line in train_lines[:k]:
                        f.write(line)
                with jsonlines.open(os.path.join(setting_dir, "dev.jsonl"), "w") as f:
                    val_rate = 11 if "10x" in args.mode else 2
                    for line in train_lines[k : k * val_rate]:
                        f.write(line)
            else:
                # Get label list for balanced sampling
                label_list = {}
                for line in train_lines:
                    label = line["label"]
                    if label not in label_list:
                        label_list[label] = [line]
                    else:
                        label_list[label].append(line)
                with jsonlines.open(os.path.join(setting_dir, "train.jsonl"), "w") as f:
                    for label in label_list:
                        for line in label_list[label][:k]:
                            f.write(line)
                with jsonlines.open(os.path.join(setting_dir, "dev.jsonl"), "w") as f:
                    for label in label_list:
                        val_rate = 11 if "10x" in args.mode else 2
                        for line in label_list[label][k : k * val_rate]:
                            f.write(line)


if __name__ == "__main__":
    main()
