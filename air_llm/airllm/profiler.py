"""Simple profiler for layer-by-layer timing."""

import torch
from collections import defaultdict


class LayeredProfiler:
    __slots__ = ('profiling_time_dict', 'print_memory', 'min_free_mem')

    def __init__(self, print_memory=False):
        self.profiling_time_dict = defaultdict(list)
        self.print_memory = print_memory
        self.min_free_mem = float('inf')

    def add_profiling_time(self, item, time_val):
        self.profiling_time_dict[item].append(time_val)
        if self.print_memory and torch.cuda.is_available():
            free = torch.cuda.mem_get_info()[0]
            self.min_free_mem = min(self.min_free_mem, free)
            print(f"free vmem @{item}: {free/2**30:.2f}GB, min: {self.min_free_mem/2**30:.2f}GB")

    def clear_profiling_time(self):
        for v in self.profiling_time_dict.values():
            v.clear()

    def print_profiling_time(self):
        for item, times in self.profiling_time_dict.items():
            print(f"total time for {item}: {sum(times):.4f}s")
