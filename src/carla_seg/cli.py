"""Small command-line interface for the portfolio's reproducible workflow."""

import argparse
import json
from pathlib import Path

import numpy as np

from .data import pair_files, load_label
from .engine import train, evaluate, predict, save_json


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description="13-class CARLA semantic segmentation with the submitted U-Net")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "evaluate", "predict", "inspect-data"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--data-dir", type=Path, required=True, help="Root containing Training/ and Validation/")
        sub.add_argument("--skip-missing", action="store_true", help="Explicitly exclude unmatched filenames and report them")
        if command == "inspect-data":
            sub.add_argument("--output", type=Path, help="Optional JSON audit destination")
            continue
        sub.add_argument("--height", type=positive_int, default=600)
        sub.add_argument("--width", type=positive_int, default=800)
        sub.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
        if command != "predict":
            sub.add_argument("--batch-size", type=positive_int, default=4)
            sub.add_argument("--ce-weight", type=float, default=0.5, help="CE weight; Dice receives 1 minus this value")
        if command == "train":
            sub.add_argument("--output-dir", type=Path, default=Path("runs/final"))
            sub.add_argument("--epochs", type=positive_int, default=100)
            sub.add_argument("--patience", type=positive_int, default=10)
            sub.add_argument("--learning-rate", type=float, default=0.0002)
            sub.add_argument("--weight-decay", type=float, default=0.01)
            sub.add_argument("--seed", type=int, default=42)
        else:
            sub.add_argument("--weights", type=Path, required=True)
            sub.add_argument("--split", choices=("Training", "Validation"), default="Validation")
            if command == "evaluate":
                sub.add_argument("--output", type=Path, default=Path("results/evaluation.json"))
                sub.add_argument("--limit", type=positive_int, help="Evaluate first N pairs; result is explicitly marked as a subset")
            else:
                sub.add_argument("--output", type=Path, default=Path("results/predictions.png"))
                sub.add_argument("--count", type=positive_int, default=3)
    return parser


def inspect_data(args):
    reports = []
    for split in ("Training", "Validation"):
        pairs, report = pair_files(args.data_dir, split, args.skip_missing)
        class_ids = set()
        for _, label_path in pairs:
            class_ids.update(np.unique(load_label(label_path)).tolist())
        report.update(label_files_checked=len(pairs), class_ids_after_255_to_0=sorted(class_ids))
        reports.append(report)
    result = {"splits": reports, "label_policy": "Red channel or grayscale IDs; 255 maps to class 0; valid IDs 0–12."}
    if args.output:
        save_json(args.output, result)
    print(json.dumps(result, indent=2))


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "height", 600) < 16 or getattr(args, "width", 800) < 16:
        parser.error("--height and --width must be at least 16")
    if not 0 <= getattr(args, "ce_weight", 0.5) <= 1:
        parser.error("--ce-weight must be between 0 and 1")
    if getattr(args, "learning_rate", 1) <= 0 or getattr(args, "weight_decay", 0) < 0:
        parser.error("Learning rate must be positive; weight decay must be nonnegative")
    try:
        {"train": train, "evaluate": evaluate, "predict": predict, "inspect-data": inspect_data}[args.command](args)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"error: {exc}\n")
