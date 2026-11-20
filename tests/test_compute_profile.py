"""Dispatch scopes must conserve counts across grouped rows and device threads."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/compute"))
from op_counter import OpCounter
from methods.mobihet_mixed_l4 import Lockstep


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device needed for native autograd workers"))])
def test_lockstep_backward_and_padding(device):
    dev = torch.device("cuda", torch.cuda.current_device()) if device == "cuda" else torch.device("cpu")
    counter = OpCounter()
    counter.phase = "r2.train"

    def work(rows):
        # Last two rows are padding. Both GEMMs still execute across all rows.
        a = torch.ones(2, rows, 3, device=dev, requires_grad=True)
        b = torch.ones(2, 3, 5, device=dev, requires_grad=True)
        with counter.scope("client.forward"):
            z = torch.bmm(a, b)
        with counter.scope("client.backward"), counter.autograd():
            z[:, :-2].sum().backward()

    with counter:
        Lockstep(2, lambda x: x).run([lambda: work(4), lambda: work(6)], dev)
    expected_forward = 2 * 2 * (4 + 6) * 3 * 5
    detail = counter.detailed_table()
    flop = {row["scope"]: row["operations"] for row in detail if row["convention"] == "formula"}
    assert flop == {"client.forward": expected_forward, "client.backward": 2 * expected_forward}
    assert sum(flop.values()) == sum(row["mm_flops"] for row in counter.table())


def test_element_estimates_exclude_copy_and_reset_scope():
    counter = OpCounter()
    counter.phase = "r2.round_other"
    with counter:
        a = torch.ones(4)
        with counter.scope("server.model_aggregation"):
            clone = a.clone()
            clone + a
        a * 2
    by_op = {row["operator"]: row for row in counter.detailed_table()}
    assert by_op["clone"]["convention"] == "element_work"
    assert by_op["add"]["convention"] == "arithmetic_estimate"
    assert by_op["add"]["scope"] == "server.model_aggregation"
    assert by_op["mul"]["scope"] == "unattributed"
