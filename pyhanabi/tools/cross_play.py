import argparse
import os
import sys
import pprint
import itertools
from collections import defaultdict
import numpy as np
import torch
import re
import json
lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
from eval import evaluate_saved_model
import common_utils


def filter_include(entries, includes):
    if not isinstance(includes, list):
        includes = [includes]
    keep = []
    for entry in entries:
        for include in includes:
            if include not in entry:
                break
        else:
            keep.append(entry)
    return keep


def filter_exclude(entries, excludes):
    if not isinstance(excludes, list):
        excludes = [excludes]
    keep = []
    for entry in entries:
        for exclude in excludes:
            if exclude in entry:
                break
        else:
            keep.append(entry)
    return keep


def cross_play(comb_string, num_game, seed):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    comb_string = comb_string.replace(',','')
    comb_string = comb_string.replace(']','')
    paths = re.findall(r'/[^"]+', comb_string)
    print('paths:', paths)
    comb = paths[0].split('+')
    print('line comb 46', comb)
    # combs = list(itertools.combinations_with_replacement(models, num_player))
    # print(combs)
    # print(len(combs))

    # folder_name = paths[0].split('+')[0].split('/')[-3]

    folder_path = '/home/mila/n/nekoeiha/scratch/final_hanabi_checkpoint/zero_shot_eval_2p'
    # folder_path = os.path.join(base_folder_path, folder_name)

    os.makedirs(folder_path, exist_ok=True)

    perfs = defaultdict(list)
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            json_file_path = os.path.join(folder_path, file_name)
            # Open and load the JSON file
            with open(json_file_path, 'r') as file:
                data = json.load(file)
            for key, value in data.items():
                if isinstance(value, list):
                    perfs[key].extend(value)  # Append list items to the existing list
                else:
                    perfs[key].append(value)  # Append single value to the list

    num_model = len(set(comb))
    score = evaluate_saved_model(comb, num_game, seed, 0, device=device)[0]
    print(score)
    print('perfs_index', num_model)
    perfs[comb_string].append(score)

    result_json_path = os.path.join(folder_path, 'result.json')

    with open(result_json_path, 'w') as file:
        json.dump(perfs, file, indent=4)

    # perfs[num_model].append(score)
    #
    # for num_model, scores in perfs.items():
    #     print(
    #         f"#model: {num_model}, #groups {len(scores)}, "
    #         f"score: {np.mean(scores):.2f} "
    #         f"+/- {np.std(scores) / np.sqrt(len(scores) - 1):.2f}"
    #     )
    # print('perfs', perfs)

def find_pth_files(directory):
    pth_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pthw"):
                file_path = os.path.join(root, file)
                pth_files.append(file_path)
    return pth_files


parser = argparse.ArgumentParser()
parser.add_argument("--root", default=None, type=str, required=True)
parser.add_argument("--num_player", default=2, type=int)
parser.add_argument("--include", default=None, type=str, nargs="+")
parser.add_argument("--exclude", default=None, type=str, nargs="+")

args = parser.parse_args()

# models = common_utils.get_all_files(args.root, "*.pthw")
# models = find_pth_files(args.root)

# print(models)
#
# if args.include is not None:
#     models = filter_include(models, args.include)
# if args.exclude is not None:
#     models = filter_exclude(models, args.exclude)

pprint.pprint(args.root)
cross_play(args.root, 1000, 1)