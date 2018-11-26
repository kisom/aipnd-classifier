#!/usr/bin/env python3

import argparse
import copy
import json
import sys
import torch

import dataset
import gym
import model
import util

log = util.get_logger()


def train_new_network(hp, data_dir, batch_size=32, print_every=50):
    m = model.Model(hp)
    log.info("instantiated model: {}".format(m))
    d = dataset.Dataset(data_dir, batch_size)
    log.info("loaded dataset: {}".format(d))
    g = gym.Gym(m, d, print_every)

    g.train()
    accuracy = g.evaluate()
    if accuracy > 0.7:
        m.save("checkpoint.dat")
        return m


def try_multiple_experiments(hp, experiment_file, data_dir, batch_size, print_every):
    hp_delta = None
    with open(experiment_file, "rt") as experiments:
        hp_delta = json.loads(experiments.read())
    for delta in hp_delta:
        log.info("running experiment: {}".format(delta))
        hp_mod = copy.copy(hp)
        hp_mod.update(delta)
        if train_new_network(hp_mod, data_dir, batch_size):
            log.info("found a viable model")
            return


def main(args):
    default_hp = model.DEFAULT_HYPER_PARAMETERS

    parser = argparse.ArgumentParser(description="Train a neural network.")
    parser.add_argument(
        "--arch",
        dest="arch",
        action="store",
        default=default_hp["architecture"],
        help="select a vision model",
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
        "--batch-size",
        dest="batch_size",
        action="store",
        default=16,
        help="training batch size",
        type=int,
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
        "--data-dir",
        dest="data_dir",
        action="store",
        default="flowers",
        help="training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        dest="learning_rate",
        action="store",
        default=0.01,
        help="training batch size",
        type=float,
    )
    parser.add_argument(
        "--experiments",
        dest="experiments",
        action="store",
        help="run experiments defined in the named file",
    )
    parser.add_argument(
        "--no-gpu", dest="no_gpu", action="store_true", help="disable GPU requirement"
    )
    args = parser.parse_args(args)

    hp = copy.copy(default_hp)
    hp["architecture"] = args.arch
    hp["epochs"] = args.epochs
    hp["learning_rate"] = args.learning_rate

    if not args.no_gpu:
        assert torch.cuda.is_available()

    if args.experiments:
        try_multiple_experiments(
            hp, args.experiments, args.data_dir, args.batch_size, args.print_every
        )
    else:
        train_new_network(hp, args.data_dir, args.batch_size, args.print_every)


if __name__ == "__main__":
    main(sys.argv[1:])
