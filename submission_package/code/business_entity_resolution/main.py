"""
End-to-end pipeline entry point.

Usage:
    python main.py --data-dir dataset --output-dir output --model-dir model

Runs, in order:
  1. src/train.py  -- trains the classifier on dataset/train/, tunes a
     decision threshold on a held-out split, saves model/classifier.joblib
     and model/config.json
  2. src/infer.py  -- generates candidates + scores + writes
     output/candidate_pairs.tsv and output/matching_results.tsv for
     dataset/test/

Run the two stages independently (`python -m src.train ...` /
`python -m src.infer ...`) if you want to re-run inference without
retraining, e.g. while experimenting with --threshold.
"""

import argparse
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--model-dir", default="model")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--no-embeddings", action="store_true")
    ap.add_argument("--skip-train", action="store_true", help="reuse an existing model dir")
    args = ap.parse_args()

    if not args.skip_train:
        train_cmd = [
            sys.executable,
            "-m",
            "src.train",
            "--data-dir",
            args.data_dir,
            "--model-out",
            args.model_dir,
            "--val-frac",
            str(args.val_frac),
        ]
        if args.no_embeddings:
            train_cmd.append("--no-embeddings")
        subprocess.run(train_cmd, check=True)

    infer_cmd = [
        sys.executable,
        "-m",
        "src.infer",
        "--data-dir",
        args.data_dir,
        "--model-dir",
        args.model_dir,
        "--output-dir",
        args.output_dir,
    ]
    subprocess.run(infer_cmd, check=True)


if __name__ == "__main__":
    main()
