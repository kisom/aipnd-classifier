#!/usr/bin/env python3

import argparse
import copy
import json
import os
import sys
import time
import torch

import dataset
import gym
import model
import util

log = util.get_logger()


def main(args):
    parser = argparse.ArgumentParser(description="Train a neural network.")
    parser.add_argument(
        "--arch",
        dest="arch",
        action="store",
        default="vgg",
        help="select a vision model",
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        action="store",
        default=16,
        help="training batch size",
        type=int,
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        help="restore from a checkpoint before training",
    )
    parser.add_argument(
        "--data_dir",
        dest="data_dir",
        action="store",
        default="flowers",
        help="training batch size",
    )
    parser.add_argument(
        "--epochs",
        dest="epochs",
        action="store",
        default=3,
        help="number of training epochs",
        type=int,
    )
    parser.add_argument(
        "--gpu", dest="gpu", action="store_true", help="enable GPU training"
    )
    parser.add_argument(
        "--learning_rate",
        dest="learning_rate",
        action="store",
        default=0.01,
        help="training batch size",
        type=float,
    )
    parser.add_argument(
        "--hidden_units",
        dest="layers",
        action="store",
        default=None,
        help="specify the hidden layers as a comma-separated list (e.g. 4096,4096)",
    )
    parser.add_argument(
        "--print-every",
        dest="print_every",
        action="store",
        default=50,
        help="number of steps to print updates",
        type=int,
    )
    parser.add_argument(
        "--save_dir",
        dest="save_dir",
        action="store",
        default=".",
        help="directory to save checkpoints",
    )
    args = parser.parse_args(args)
    data = dataset.Dataset(args.data_dir, args.batch_size)
    hp = copy.copy(model.DEFAULT_HYPER_PARAMETERS)
    hp["architecture"] = args.arch
    hp["learning_rate"] = args.learning_rate

    if args.layers:
        hp["layers"] = [int(l) for l in args.layers.split(",")]

    if args.checkpoint:
        nn = model.load(args.checkpoint)
    else:
        nn = model.Model(hp, data.class_to_idx)
    

    if args.gpu:
        device = "cuda"
    else:
        device = "cpu"
    arena = gym.Gym(nn, data, args.print_every, device)
    arena.train(args.epochs)
    accuracy, _ = arena.evaluate()
    if accuracy >= 0.7:
        checkpoint_path = checkpoint_name(args.save_dir)
        nn.save(checkpoint_path)
        print("Wrote checkpoint to {}".format(checkpoint_path))


def checkpoint_name(save_dir):
    return os.path.join(save_dir, "checkpoint-{}".format(time.strftime("%s")))


if __name__ == "__main__":
    main(sys.argv[1:])
