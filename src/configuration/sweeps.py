"""Expand ordered sweep fields; order determines scenario seed assignment."""

from itertools import product


def expand_sweeps(fields):
    names = [name for name, _ in fields]
    values = [values for _, values in fields]
    return [dict(zip(names, row)) for row in product(*values)]
