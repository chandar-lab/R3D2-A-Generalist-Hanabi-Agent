# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import sys
import numpy as np
import torch
import common_utils
import glob
import os
import wandb
import json

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
from eval import evaluate_saved_model


# main program
parser = argparse.ArgumentParser()
parser.add_argument("--weight1", default=None, type=str)
parser.add_argument("--weight2", default=None, type=str)
parser.add_argument(
    "--weights",
    type=lambda x: x.split(","),
    default=None,
    help="Comma-separated list of items",
)
parser.add_argument("--num_player", default=2, type=int)
parser.add_argument("--num_alternative_player", default=1, type=int)
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--bomb", default=0, type=int)
parser.add_argument("--num_game", default=10, type=int)
parser.add_argument(
    "--num_run",
    default=1,
    type=int,
    help="num of {num_game} you want to run, i.e. num_run=2 means 2*num_game",
)
parser.add_argument("--overwrite1", default=None, type=str)
parser.add_argument("--overwrite2", default=None, type=str)

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
if args.weights is not None:
    weight_files = args.weights
    epoch_number = int(
        args.weights[-1].split("/")[-1].split(".")[0].replace("epoch", "")
    )
    wandb_id = "_".join(args.weights[0].split("/")[-6:]) + "_".join(
        args.weights[-1].split("/")[-6:-1]
    )
else:
    weight_files = [
        args.weight1 for _ in range(args.num_player - args.num_alternative_player)
    ] + [args.weight2] * args.num_alternative_player
    wandb_id = "_".join(args.weight1[0].split("/")[-6:]) + "_".join(
        args.weight2[-1].split("/")[-6:-1]
    )
    epoch_number = int(args.weight1.split("/")[-1].split(".")[0].replace("epoch", ""))

print(weight_files)
print(f"wandb id: {wandb_id}")
wandb.init(
    project="R3D2-eval",
    entity="default-entity",
    config=args,
    resume="allow",
    id=wandb_id,
    mode="disabled",
)  # ,

_, _, perfect, scores, actors, _ = evaluate_saved_model(
    weight_files,
    args.num_game,
    args.seed,
    args.bomb,
    num_run=args.num_run,
    device=device,
    overwrites=None,
)
non_zero_scores = [s for s in scores if s > 0]
# print(args.weight1 , args.weight2)
print(f"non zero mean: {np.mean(non_zero_scores):.3f}")
wandb.log(
    {
        f"{args.num_player}/epoch": epoch_number,
        f"{args.num_player}/score": np.mean(scores),
        f"{args.num_player}/perfect": perfect,
    }
)

wandb.finish()
