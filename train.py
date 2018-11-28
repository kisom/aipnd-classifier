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
    d = dataset.Dataset(data_dir, batch_size)
    log.info("loaded dataset: {}".format(d))
    m = model.Model(hp, labels, d.class_to_idx)
    log.info("instantiated model: {}".format(m))

    g = gym.Gym(m, d, print_every)

    g.train()
    accuracy, _ = g.evaluate()
    if accuracy > 0.7:
        m.save("checkpoint.dat")
    return accuracy


def train_from_checkpoint(path, epochs, data_dir, batch_size, print_every):
    m = model.Model.load(path)
    m.hyper_params["epochs"] = epochs
    log.info("restored model: {}".format(m))
    d = dataset.Dataset(data_dir, batch_size)
    log.info("loaded dataset: {}".format(d))
    g = gym.Gym(m, d, print_every)
    pre_acc = g.evaluate()

    g.train(max_stalls=5)
    post_acc = g.evaluate()

    if post_acc > pre_acc:
        improvement = (post_acc - pre_acc) / pre_acc
        log.info(
            "model improved with training by {:0.3}% (new accuracy {:0.4}".format(
                improvement, post_acc
            )
        )
        m.save("checkpoint.dat")


def try_multiple_experiments(hp, experiment_file, data_dir, batch_size, print_every):
    hp_delta = None
    results = {}
    with open(experiment_file, "rt") as experiments:
        hp_delta = json.loads(experiments.read())
    for delta in hp_delta:
        log.info("running experiment: {}".format(delta))
        hp_mod = copy.copy(hp)
        hp_mod.update(delta)
        accuracy = train_new_network(hp_mod, data_dir, batch_size)
        if accuracy > 0.7:
            log.info("found a viable model")
        results[str(delta)] = accuracy
    with open("results.txt", "a") as results_out:
        results_out.write(json.dumps(results))


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
        "--batch-size",
        dest="batch_size",
        action="store",
        default=16,
        help="training batch size",
        type=int,
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        action="store",
        help="resume training from the checkpoint",
    )
    parser.add_argument(
        "--data-dir",
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
        "--experiments",
        dest="experiments",
        action="store",
        help="run experiments defined in the named file",
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
        "--no-gpu", dest="no_gpu", action="store_true", help="disable GPU requirement"
    )
    parser.add_argument(
        "--print-every",
        dest="print_every",
        action="store",
        default=50,
        help="number of steps to print updates",
        type=int,
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
    elif args.checkpoint:
        train_from_checkpoint(
            args.checkpoint,
            args.epochs,
            args.data_dir,
            args.batch_size,
            args.print_every,
        )
    else:
        train_new_network(hp, args.data_dir, args.batch_size, args.print_every)


if __name__ == "__main__":
    labels = json.loads(open('cat_to_name.json', 'rt').read())
    main(sys.argv[1:])
