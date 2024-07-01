import itertools
import numpy as np
import torch
from typing import Dict, List, Tuple, Iterable


def generate_hparam_configs(base_config:Dict, hparam_ranges:Dict) -> Tuple[List[Dict], List[str]]:
    """
    Generate a list of hyperparameter configurations for hparam sweeping

    :param base_config (Dict): base configuration dictionary
    :param hparam_ranges (Dict): dictionary mapping hyperparameter names to lists of values to sweep over
    :return (Tuple[List[Dict], List[str]]): list of hyperparameter configurations and swept parameter names
    """

    keys, values = zip(*hparam_ranges.items())
    hparam_configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    swept_params = list(hparam_ranges.keys())

    new_configs = []
    for hparam_config in hparam_configurations:
        new_config = base_config.copy()
        new_config.update(hparam_config)
        new_configs.append(new_config)

    return new_configs, swept_params


def grid_search(num_samples: int, min: float = None, max: float = None, **kwargs)->Iterable:
    """ Implement this method to set hparam range over a grid of hyperparameters.
    :param num_samples (int): number of samples making up the grid
    :param min (float): minimum value for the allowed range to sweep over
    :param max (float): maximum value for the allowed range to sweep over
    :param kwargs: additional keyword arguments to parametrise the grid.
    :return (Iterable): tensor/array/list/etc... of values to sweep over

    Example use: hparam_ranges['batch_size'] = grid_search(64, 512, 6, log=True)
    """
    log = kwargs.get('log', False)
    if log:
        values = torch.logspace(np.log10(min), np.log10(max), num_samples)
    else:
        values = torch.linspace(min, max, num_samples)
    return values


def random_search(num_samples: int, distribution: str, min: float=None, max: float=None, **kwargs) -> Iterable:
    """ Implement this method to sweep via random search, sampling from a given distribution.
    :param num_samples (int): number of samples to take from the distribution
    :param distribution (str): name of the distribution to sample from
        (you can instantiate the distribution using torch.distributions, numpy.random, or else).
    :param min (float): minimum value for the allowed range to sweep over (for continuous distributions)
    :param max (float): maximum value for the allowed range to sweep over (for continuous distributions)
    :param kwargs: additional keyword arguments to parametrise the distribution.

    Example use: hparam_ranges['lr'] = random_search(1e-6, 1e-1, 10, distribution='exponential', lambda=0.1)
    """
    values = torch.zeros(num_samples)

    if distribution == 'exponential':
        dist = torch.distributions.exponential.Exponential(kwargs['lambda'])
        values = dist.sample((num_samples,))
    elif distribution == 'uniform':
        values = torch.rand(num_samples) * (max - min) + min
    elif distribution == 'normal':
        dist = torch.distributions.normal.Normal(kwargs['loc'], kwargs['scale'])
        values = dist.sample((num_samples,))
    elif distribution == 'lognormal':
        dist = torch.distributions.log_normal.LogNormal(kwargs['mean'], kwargs['std'])
        values = dist.sample((num_samples,))
    elif distribution == 'geometric':
        dist = torch.distributions.geometric.Geometric(kwargs['p'])
        values = dist.sample((num_samples,))
    elif distribution == 'beta':
        dist = torch.distributions.beta.Beta(kwargs['alpha'], kwargs['beta'])
        values = dist.sample((num_samples,))
    elif distribution == 'binomial':
        dist = torch.distributions.binomial.Binomial(kwargs['total_count'], kwargs['probs'])
        values = dist.sample((num_samples,))
    elif distribution == 'poisson':
        dist = torch.distributions.poisson.Poisson(kwargs['rate'])
        values = dist.sample((num_samples,))
    elif distribution == 'gamma':
        dist = torch.distributions.gamma.Gamma(kwargs['concentration'], kwargs['rate'])
        values = dist.sample((num_samples,))
    elif distribution == 'dirichlet':
        dist = torch.distributions.dirichlet.Dirichlet(kwargs['concentration'])
        values = dist.sample((num_samples,))
    elif distribution == 'multinomial':
        dist = torch.distributions.multinomial.Multinomial(kwargs['total_count'], kwargs['probs'])
        values = dist.sample((num_samples,))
    return values
