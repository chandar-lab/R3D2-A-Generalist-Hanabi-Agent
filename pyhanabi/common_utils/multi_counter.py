from collections import defaultdict, Counter
from datetime import datetime
import sys

class ValueStats:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def feed(self, v):
        self.summation += v
        if v > self.max_value:
            self.max_value = v
            self.max_idx = self.counter
        if v < self.min_value:
            self.min_value = v
            self.min_idx = self.counter

        self.counter += 1

    def mean(self):
        if self.counter == 0:
            print("Counter %s is 0" % self.name)
            assert False
        return self.summation / self.counter

    def summary(self, info=None,wb_log=False):
        info = "" if info is None else info
        name = "" if self.name is None else self.name
        if self.counter > 0 and wb_log==False:
            return "%s%s[%4d]: avg: %8.4f, min: %8.4f[%4d], max: %8.4f[%4d]" % (
                info,
                name,
                self.counter,
                self.summation / self.counter,
                self.min_value,
                self.min_idx,
                self.max_value,
                self.max_idx,
            )
        elif self.counter>0 and wb_log==True:
            return self.summation / self.counter

        else:
            return "%s%s[0]" % (info, name)

    def reset(self):
        self.counter = 0
        self.summation = 0.0
        self.max_value = -1e38
        self.min_value = 1e38
        self.max_idx = None
        self.min_idx = None


class MultiCounter:
    def __init__(self, root, verbose=False):
        self.last_time = None
        self.verbose = verbose
        self.stats = defaultdict(lambda: ValueStats())
        self.total_count = 0
        self.max_key_len = 0
        # TODO: add dump

    def __getitem__(self, key):
        if len(key) > self.max_key_len:
            self.max_key_len = len(key)

        return self.stats[key]

    def inc(self, key):
        if self.verbose:
            print("[MultiCounter]: %s" % key)
        self.counts[key] += 1
        self.total_count += 1
        if self.last_time is None:
            self.last_time = datetime.now()

    def ret_dict(self,global_counter):
        metric_wandb_di={}
        for k, v in self.stats.items():
            info = str(global_counter) + ":" + k
            avg =v.summary(info=info.ljust(self.max_key_len + 4), wb_log=True)
            metric_wandb_di[k] = avg
        return metric_wandb_di


    def reset(self):
        for k in self.stats.keys():
            self.stats[k].reset()

        self.counts = Counter()
        self.total_count = 0
        self.last_time = datetime.now()

    def summary(self, global_counter):
        assert self.last_time is not None
        time_elapsed = (datetime.now() - self.last_time).total_seconds()
        print("[%d] Time spent = %.2f s" % (global_counter, time_elapsed))

        for key, count in self.counts.items():
            print("%s: %d/%d" % (key, count, self.total_count))

        for k, v in self.stats.items():
            info = str(global_counter) + ":" + k
            print(v.summary(info=info.ljust(self.max_key_len + 4)))
