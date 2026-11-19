"""Histogram sampler with the dataset's missing-flag => zero-imputed-value constraint."""
import json
from pathlib import Path

import torch

from methods.mobihet_stats_hist import Method as Hist


ORDER = json.loads((Path(__file__).resolve().parents[1] / 'knowledge/input_order.json').read_text())['inputs']
PAIRS = [(ORDER.index(name), ORDER.index(name[:-4])) for name in ORDER if name.endswith('_nan')]


def enforce_missing(x):
    result = x.clone()
    for flag, value in PAIRS:
        result[..., value] = torch.where(x[..., flag] == 1, 0., x[..., value])
    return result


class Method(Hist):
    name = 'mobihet_stats_histmask'

    def _gaussian_rows(self, z):
        return enforce_missing(super()._gaussian_rows(z))
