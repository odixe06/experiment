"""Extended operation counter, one convention for every method (docs/research/compute-harness.md, owner 19/11/2026).

Every aten op dispatched while a phase is set is counted once, forward and backward alike (TorchDispatchMode is
propagated to the autograd threads):
  mm     ops with a torch.utils.flop_counter formula (mm/bmm/addmm/baddbmm/conv/SDPA ...): that formula's FLOPs, the
         convention of the earlier cost.py tables (2 x multiply-adds);
  other  every other op except views/aliases/metadata and uninitialised allocations: one op per element it produces,
         or per element it reads for reductions (sum/mean/amax/var/norm/topk/bincount/histc ...); indexing reads
         count the rows they gather, scatters the values they write. Data movement (copy/cat/cast) counts as an op per
         element. Not FLOPs in the strict sense: report mm and other separately, total = mm + other.
Ops are counted on whatever device they run (server code on CPU included); device is recorded alongside.
"""
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from torch.utils.flop_counter import flop_registry

aten = torch.ops.aten
FREE = {"empty", "empty_like", "empty_strided", "new_empty", "new_empty_strided", "detach", "alias", "lift_fresh",
        "_local_scalar_dense", "set_", "resize_", "is_same_size", "_has_compatible_shallow_copy_type", "record_stream",
        "sym_size", "sym_stride", "sym_numel", "sym_storage_offset", "is_nonzero", "_to_dense"}
READS_ALL = {"sum", "mean", "amax", "amin", "max", "min", "prod", "var", "std", "var_mean", "std_mean", "norm",
             "linalg_vector_norm", "logsumexp", "argmax", "argmin", "any", "all", "topk", "kthvalue", "median",
             "bincount", "histc", "histogram", "unique", "_unique2", "unique_consecutive", "count_nonzero", "nonzero",
             "nll_loss_forward", "nll_loss2d_forward", "native_layer_norm", "native_batch_norm", "searchsorted",
             "bucketize", "dist", "cdist", "_cdist_forward", "equal", "aminmax", "nansum", "cummax", "cummin"}
WRITES = {"index_put", "index_put_", "_index_put_impl_", "scatter", "scatter_", "scatter_add", "scatter_add_",
          "scatter_reduce", "scatter_reduce_", "index_add", "index_add_", "index_copy", "index_copy_", "masked_scatter",
          "masked_scatter_", "index_fill", "index_fill_"}

# Unit-per-element arithmetic approximation reported independently of the registry.
# Indexing, casts, copies, allocation, RNG, comparisons and sorting are excluded.
ARITHMETIC = {
    "add", "sub", "rsub", "mul", "div", "true_divide", "neg", "reciprocal", "addcdiv", "addcmul",
    "abs", "clamp", "clamp_min", "clamp_max", "square", "pow", "sqrt", "rsqrt", "exp", "expm1",
    "log", "log1p", "erf", "erfinv", "sin", "cos", "tanh", "sigmoid", "silu", "relu", "gelu",
    "silu_backward", "gelu_backward", "threshold_backward", "tanh_backward",
    "sigmoid_backward", "round", "floor", "ceil", "sum", "mean", "prod", "var", "std",
    "var_mean", "std_mean", "norm", "linalg_vector_norm", "logsumexp", "nansum", "cumsum",
    "_softmax", "_softmax_backward_data", "_log_softmax", "_log_softmax_backward_data",
    "native_layer_norm", "native_layer_norm_backward", "native_batch_norm", "native_batch_norm_backward",
    "nll_loss_forward", "nll_loss_backward", "scatter_add", "index_add", "lerp",
}


def _numel(tree):
    return [t.numel() for t in tree_flatten(tree)[0] if isinstance(t, torch.Tensor)]


def _device(tree):
    for t in tree_flatten(tree)[0]:
        if isinstance(t, torch.Tensor):
            return t.device.type
    return "cpu"


class OpCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self._phase = ContextVar("compute_phase", default=None)
        self._labels = ContextVar("compute_labels", default=())
        self._autograd_context = None
        self.details = defaultdict(int)   # (phase, scope, device, operator, convention)
        self.mm = defaultdict(int)          # (phase, device) -> FLOPs
        self.other = defaultdict(int)       # (phase, device) -> element ops

    @property
    def phase(self):
        return self._phase.get()

    @phase.setter
    def phase(self, phase):
        self._phase.set(phase)

    @contextmanager
    def scope(self, label):
        token = self._labels.set((*self._labels.get(), label))
        try:
            yield
        finally:
            self._labels.reset(token)

    @contextmanager
    def autograd(self):
        """Bridge autograd's native device threads, which inherit dispatch modes only."""
        if self._autograd_context is not None:
            raise RuntimeError("operation profiling requires serialized backward calls")
        self._autograd_context = (self.phase, self._labels.get())
        try:
            yield
        finally:
            self._autograd_context = None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        ph, labels = self.phase, self._labels.get()
        if ph is None and self._autograd_context is not None:
            ph, labels = self._autograd_context
        if ph is None:
            return out
        pkt = func._overloadpacket
        device = _device(out) if _numel(out) else _device(args)
        key = (ph, device)
        name = pkt.__name__
        scope = "/".join(labels) or "unattributed"
        detail = (ph, scope, device, name)
        if pkt in flop_registry:
            n = int(flop_registry[pkt](*args, **kwargs, out_val=out))
            self.mm[key] += n
            self.details[*detail, "formula"] += n
            return out
        if func.is_view or name in FREE:
            return out
        if name in WRITES:                                            # values written: the source operand
            src = kwargs.get("values", kwargs.get("src", kwargs.get("source")))
            if src is None:
                src = args[-1] if len(args) > 2 and isinstance(args[-1], torch.Tensor) else None
            n = src.numel() if isinstance(src, torch.Tensor) else max(_numel(out) or [0])
        elif name in READS_ALL:
            n = max(_numel((args, kwargs)) or [0])
        else:
            n = sum(_numel(out))
        self.other[key] += int(n)
        category = "arithmetic_estimate" if name.rstrip("_") in ARITHMETIC or name in ARITHMETIC else "element_work"
        self.details[*detail, category] += int(n)
        return out

    def table(self):
        keys = sorted(set(self.mm) | set(self.other))
        return [{"phase": p, "device": d, "mm_flops": self.mm.get((p, d), 0), "other_ops": self.other.get((p, d), 0)}
                for p, d in keys]

    def detailed_table(self):
        return [{"phase": phase, "scope": scope, "device": device, "operator": operator,
                 "convention": convention, "operations": n}
                for (phase, scope, device, operator, convention), n in sorted(self.details.items())]
