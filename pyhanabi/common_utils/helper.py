import os
import random
import psutil
import argparse
import sys
import yaml

import numpy as np
import torch
from torch import nn


def maybe_load_config(args):
    if args.config is None:
        return args

    dict_args = vars(args)
    set_flags = [k[2:] for k in sys.argv[1:]]

    config = yaml.safe_load(open(args.config, "r"))
    for k, v in config.items():
        assert k in dict_args, f"Invalid argument [{k}] from config at {args.config}"
        if k not in set_flags:
            # only use config val if user does not set it
            dict_args[k] = v

    new_args = argparse.Namespace()
    for k, v in dict_args.items():
        setattr(new_args, k, v)
    return new_args


def to_device(data, device, detach=False):
    if isinstance(data, torch.Tensor):
        if detach:
            return data.to(device).detach()
        else:
            return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]


def get_all_files(root, file_extension, contain=None):
    files = []
    for folder, _, fs in os.walk(root):
        for f in fs:
            if file_extension is not None:
                if f.endswith(file_extension):
                    if contain is None or contain in os.path.join(folder, f):
                        files.append(os.path.join(folder, f))
            else:
                if contain in f:
                    files.append(os.path.join(folder, f))
    return files


def flatten(s):
    if s == []:
        return s
    if isinstance(s[0], list):
        return flatten(s[0]) + flatten(s[1:])
    return s[:1] + flatten(s[1:])


def moving_average(data, period):
    if period == 1:
        return data
    # padding
    left_pad = [data[0] for _ in range(period // 2)]
    if period == 2:
        right_pad = []
    else:
        right_pad = data[-(period - 1) // 2 :]
    data = left_pad + data + right_pad
    weights = np.ones(period) / period
    return np.convolve(data, weights, mode="valid")


def mem2str(num_bytes):
    assert num_bytes >= 0
    if num_bytes >= 2**30:  # GB
        val = float(num_bytes) / (2**30)
        result = "%.3f GB" % val
    elif num_bytes >= 2**20:  # MB
        val = float(num_bytes) / (2**20)
        result = "%.3f MB" % val
    elif num_bytes >= 2**10:  # KB
        val = float(num_bytes) / (2**10)
        result = "%.3f KB" % val
    else:
        result = "%d bytes" % num_bytes
    return result


def sec2str(seconds):
    seconds = int(seconds)
    hour = seconds // 3600
    seconds = seconds % (24 * 3600)
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)


def num2str(n) -> str:
    if n < 1e3:
        s = str(n)
        unit = ""
    elif n < 1e6:
        n /= 1e3
        s = "%.2f" % n
        unit = "K"
    else:
        n /= 1e6
        s = "%.2f" % n
        unit = "M"

    s = s.rstrip("0").rstrip(".")
    return s + unit


def get_mem_usage(msg=""):
    mem = psutil.virtual_memory()
    process = psutil.Process(os.getpid())
    used = mem2str(process.memory_info().rss)
    return f"Mem info{msg}: used: {used}, avail: {mem2str(mem.available)}, total: {(mem2str(mem.total))}"


def flatten_first2dim(batch):
    if isinstance(batch, torch.Tensor):
        size = batch.size()[2:]
        batch = batch.view(-1, *size)
        return batch
    elif isinstance(batch, dict):
        return {key: flatten_first2dim(batch[key]) for key in batch}
    else:
        assert False, "unsupported type: %s" % type(batch)


def _tensor_slice(t, dim, b, e):
    if dim == 0:
        return t[b:e]
    elif dim == 1:
        return t[:, b:e]
    elif dim == 2:
        return t[:, :, b:e]
    else:
        raise ValueError("unsupported %d in tensor_slice" % dim)


def tensor_slice(t, dim, b, e):
    if isinstance(t, dict):
        return {key: tensor_slice(t[key], dim, b, e) for key in t}
    elif isinstance(t, torch.Tensor):
        return _tensor_slice(t, dim, b, e).contiguous()
    else:
        assert False, "Error: unsupported type: %s" % (type(t))


def tensor_index(t, dim, i):
    if isinstance(t, dict):
        return {key: tensor_index(t[key], dim, i) for key in t}
    elif isinstance(t, torch.Tensor):
        return _tensor_slice(t, dim, i, i + 1).squeeze(dim).contiguous()
    else:
        assert False, "Error: unsupported type: %s" % (type(t))


def one_hot(x, n):
    assert x.dim() == 2 and x.size(1) == 1
    one_hot_x = torch.zeros(x.size(0), n, device=x.device)
    one_hot_x.scatter_(1, x, 1)
    return one_hot_x


def set_all_seeds(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed + 1)
    torch.manual_seed(rand_seed + 2)
    torch.cuda.manual_seed(rand_seed + 3)


def weights_init(m):
    """custom weights initialization"""
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        # nn.init.kaiming_normal(m.weight.data)
        nn.init.orthogonal_(m.weight.data)
    else:
        print("%s is not custom-initialized." % m.__class__)


def init_net(net, net_file):
    if net_file:
        net.load_state_dict(torch.load(net_file))
    else:
        net.apply(weights_init)


def count_output_size(input_shape, model):
    fake_input = torch.FloatTensor(*input_shape)
    output_size = model.forward(fake_input).view(-1).size()[0]
    return output_size
