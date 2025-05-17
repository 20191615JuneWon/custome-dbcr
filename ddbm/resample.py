from abc import ABC, abstractmethod

import numpy as np
import torch as th
from scipy.stats import norm
import torch.distributed as dist


def create_named_schedule_sampler(name, diffusion):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "real-uniform":
        return RealUniformSampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")





class RealUniformSampler:
    def __init__(self, diffusion):
        self.sigma_max = diffusion.sigma_max
        self.sigma_min = diffusion.sigma_min

    def sample(self, batch_size, device):
        ts = th.rand(batch_size).to(device) * (self.sigma_max - self.sigma_min) + self.sigma_min
        return ts, th.ones_like(ts)


