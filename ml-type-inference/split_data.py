#!/usr/bin/env python3
"""Module for splitting data into subsets"""

from argparse import ArgumentParser, Namespace
from itertools import dropwhile
import os
import random
from typing import (Dict, Iterable, List, Mapping,
                    Optional, Sequence, Tuple, TypeVar)


T = TypeVar('T')  # pylint: disable=C0103


def get_next_project(
        projects: Sequence[Tuple[str, float]],
        upper_bound: float) -> Optional[Tuple[str, float]]:
    """Returns a random project which does not violate upper_bound"""
    try:
        allowed: Sequence[Tuple[str, float]]\
            = dropwhile(lambda t: t[1] > upper_bound, projects)
        return random.choice(allowed)  # nosec
    except IndexError:
        return None


def get_split(
        requested: Mapping[T, Tuple[float, float]],
        projects_iter: Iterable[Tuple[str, int]]) -> Optional[
            Dict[T, List[str]]]:
    """Tries to split projects into requested disjoint subsets"""
    projects: List[Tuple[str, int]] = sorted(projects_iter,
                                             key=lambda t: t[1],
                                             reverse=True)
    total: float = sum(n for _, n in projects)
    projects_normalised: List[Tuple[str, float]] = [(p, n / total)
                                                    for p, n in projects]
    result = {}
    groups: List[Tuple[T, Tuple[float, float]]]\
        = sorted(requested.items(), key=lambda t: t[1][1])
    for subset, (low, high) in groups:
        result[subset] = []
        while low > 0:
            found: Optional[Tuple[str, float]]\
                = get_next_project(projects_normalised, high)
            if found is not None:
                # pylint: disable=E0633
                proj, num = found  # type: str, float
                low -= num
                high -= num
                projects.remove(proj)
                result[subset].append(proj)
            else:
                return None
    return result


def count_annots(proj_dir: str) -> int:
    """Count number of identifier/type pairs in project"""
    count = 0
    for root, _, files in os.walk(proj_dir):  # type: str, List[str], List[str]
        for filename in filter(lambda f: os.path.splitext(f)[1] == '.csv',
                               files):  # type: str
            # pylint: disable=C0103
            with open(os.path.join(root, filename)) as f:
                count += len(f.readlines())
    return count


def get_projects(data_dir: str) -> List[Tuple[str, int]]:
    """Get projects from directory with number of type annotations in each"""
    return [(proj, count_annots(os.path.join(data_dir, proj)))
            for proj in os.listdir(data_dir)]


if __name__ == "__main__":
    PARSER: ArgumentParser = ArgumentParser(
        description='Download python repositories from github')
    PARSER.add_argument('datadir',
                        help='directory with projects')
    ARGS: Namespace = PARSER.parse_args()

    print(get_projects(ARGS.datadir))
