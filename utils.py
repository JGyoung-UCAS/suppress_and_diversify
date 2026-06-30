import copy
import datetime
import errno
import hashlib
import os
import math
import time
from collections import defaultdict, deque, OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)

def set_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[List[type]] = None,
    custom_keys_weight_decay: Optional[List[Tuple[str, float]]] = None,
    ):
    if not norm_classes:
        norm_classes = [
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
    }
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
    }
    custom_keys = []
    if custom_keys_weight_decay is not None:
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False
            for key in custom_keys:
                target_name = f"{prefix}.{name}" if prefix != "" and "." in key else name
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break
            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append({"params": params[key], "weight_decay": params_weight_decay[key]})
    return param_groups

class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU).
    Heavily based on 'torch.utils.data.DistributedSampler'.

    This is borrowed from the DeiT Repo:
    https://github.com/facebookresearch/deit/blob/main/samplers.py
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, repetitions=3):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available!")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available!")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * float(repetitions) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle
        self.seed = seed
        self.repetitions = repetitions

    def __iter__(self):
        if self.shuffle:
            # Deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(self.repetitions)]
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[: self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch