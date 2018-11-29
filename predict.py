#!/usr/bin/env python3
"""
predict contains utilities for predicting answers from a network,
e.g. running inference in the real world.
"""

import argparse
import json
import sys

import model


def main(args):
    parser = argparse.ArgumentParser(description="Train a neural network.")
    parser.add_argument(
        "--category_names",
        dest="labels",
        action="store",
        default="cat_to_name.json",
        help="JSON file mapping categories to names",
    )
    # Note: if this was passed after checkpoint, we could run the classifier on
    # multiple labels.
    parser.add_argument("input", action="store", help="image to process")
    parser.add_argument("checkpoint", action="store", help="path to saved checkpoint")
    parser.add_argument(
        "--gpu", dest="gpu", action="store_true", help="enable GPU training"
    )
    parser.add_argument(
        "--top_k",
        dest="topk",
        action="store",
        default="3",
        help="number of results to return",
        type=int,
    )
    args = parser.parse_args(args)

    with open(args.labels) as label_file:
        labels = json.loads(label_file.read())

    m = model.load(args.checkpoint)
    if args.gpu:
        m.gpu()

    predictions = model.recognize_path(m, args.input, args.topk, labels)
    for label, confidence in predictions.items():
        print("{}: {:0.5}%".format(label, confidence))


if __name__ == "__main__":
    main(sys.argv[1:])
