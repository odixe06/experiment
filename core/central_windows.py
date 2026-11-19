"""A predecessor index aligned to data.build_npy; gather ten messages only for the requested batch."""
import numpy as np
import pyarrow.parquet as pq
import torch

from core import data


def predecessors(client, sender, time, message, folds):
    # Stable ties follow message_id then original order. Only strictly earlier timestamps enter history.
    order = np.lexsort((message, time, sender, client, folds))
    prev = np.full((len(order), 2), -1, np.int32)
    if len(order) <= 1:
        return prev
    a, b = order[:-1], order[1:]
    same = (folds[a] == folds[b]) & (client[a] == client[b]) & (sender[a] == sender[b])
    # For each timestamp group use the last row of the preceding timestamp.
    starts = np.r_[True, ~same | (time[a] != time[b])]
    boundary = np.flatnonzero(starts)
    parent = np.full(len(boundary), -1, np.int32)
    contiguous = same[boundary[1:] - 1]
    parent[1:][contiguous] = order[boundary[1:][contiguous] - 1]
    prev[order, 0] = np.repeat(parent, np.diff(np.r_[boundary, len(order)]))
    prev[b[same], 1] = a[same]
    return prev


def build_index(root, split, cache):
    names = ["client_id", "sender_id", "rcv_time_ns", "message_id", "fold", "label"]
    pieces = [[] for _ in names]
    for file in data.files(root, split):
        table = pq.ParquetFile(file).read(columns=names)
        for i, name in enumerate(names):
            v = table[name].to_numpy(zero_copy_only=False)
            if name == "fold":
                v = np.where(v == "train", 0, np.where(v == "val", 1, 2)).astype(np.int8)
            pieces[i].append(v)
    columns = []
    for part in pieces:
        columns.append(np.concatenate(part))
        part.clear()
    client, sender, time, message, folds, labels = columns
    order = np.argsort(client, kind="stable")
    assert np.array_equal(labels[order], np.load(cache / f"{split}_Y.npy", mmap_mode="r"))
    assert np.array_equal(folds[order], np.load(cache / f"{split}_F.npy", mmap_mode="r"))
    prev = predecessors(client[order], sender[order], time[order], message[order], folds[order])
    np.save(cache / f"{split}_prev.npy", prev)


def batch_input(x, indices, prev=None):
    if prev is None:
        return x[indices]
    history = [indices]
    for step in range(9):
        before = prev[indices.clamp_min(0), int(step > 0)].long()
        indices = torch.where(indices >= 0, before, -1)
        history.append(indices)
    rows = torch.stack(history[::-1], 1)
    return x[rows.clamp_min(0)].masked_fill_((rows < 0).unsqueeze(-1), 0)
