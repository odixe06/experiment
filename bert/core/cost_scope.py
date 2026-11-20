"""Optional operation-count annotations. The compute harness installs the callback."""
from contextlib import nullcontext


def scope(label):
    return nullcontext()
