#!/usr/bin/env python3
"""Module for splitting data into subsets"""

from argparse import ArgumentParser, Namespace
from itertools import dropwhile
import os
from queue import PriorityQueue
import random
from typing import (Any, Dict, Iterable, List, Mapping,  # noqa
                    Optional, Sequence, TextIO, Tuple, TypeVar)

import util as myutil

T = TypeVar('T')  # pylint: disable=C0103


def get_next_project(
        projects: Sequence[Tuple[str, float]],
        upper_bound: float) -> Optional[Tuple[str, float]]:
    """Returns a random project which does not violate upper_bound"""
    try:
        allowed: Sequence[Tuple[str, float]]\
            = tuple(dropwhile(lambda t: t[1] > upper_bound, projects))
        return random.choice(allowed)  # nosec
    except IndexError:
        return None


def with_added_remaining(
        sofar: Mapping[T, Tuple[float, List[str]]],
        projects_normalised: List[Tuple[str, float]],
        groups: Mapping[T, Tuple[float, float, float]]) -> \
        Optional[Dict[T, Tuple[float, List[str]]]]:
    """TODO"""
    result: Dict[T, Tuple[float, List[str]]] = dict(sofar)
    # pylint: disable=E1136
    queue: PriorityQueue[Tuple[float, T]] = PriorityQueue()
    for grp, (_, goal, __) in groups.items():\
            # type: T, Tuple[Any, float, Any]
        queue.put((abs(sofar[grp][0] - goal), grp))
    while not queue.empty():
        group: T = queue.get_nowait()[1]
        (g_size, g_list) = result[group]
        found: Optional[Tuple[str, float]]\
            = get_next_project(projects_normalised,
                               groups[group][2] - g_size)
        if found is not None:
            # pylint: disable=E0633
            proj, num = found  # type: str, float
            result[group] = (g_size + num, g_list + [proj])
            projects_normalised.remove(found)
            queue.put_nowait((abs(result[group][0] - groups[group][1]), group))
    return result if not projects_normalised else None


# TODO: fix
# pylint: disable=R0914
def get_split(
        requested: Mapping[T, Tuple[float, float, float]],
        projects_iter: Iterable[Tuple[str, int]]) -> \
            Optional[Dict[T, Tuple[float, List[str]]]]:  # noqa
    """Tries to split projects into requested disjoint subsets"""
    projects: List[Tuple[str, int]] = sorted(projects_iter,
                                             key=lambda t: t[1],
                                             reverse=True)
    total: float = sum(n for _, n in projects)
    projects_normalised: List[Tuple[str, float]] = [(p, n / total)
                                                    for p, n in projects]
    result: Dict[T, Tuple[float, List[str]]] = {}
    groups: List[Tuple[T, Tuple[float, float, float]]]\
        = sorted(requested.items(), key=lambda t: t[1][2])
    for subset, (low, _, high) in groups:\
            # type: T, Tuple[float, float, float]
        group_projs: List[str] = []
        group_size: float = 0
        while group_size < low:
            found: Optional[Tuple[str, float]]\
                = get_next_project(projects_normalised, high - group_size)
            if found is not None:
                # pylint: disable=E0633
                proj, num = found  # type: str, float
                group_size += num
                group_projs.append(proj)
                projects_normalised.remove(found)
            else:
                return None
        result[subset] = (group_size, group_projs)
    return with_added_remaining(result, projects_normalised, requested)


def count_annots(proj_dir: str) -> int:
    """Count number of identifier/type pairs in project"""
    count = 0
    for root, _, files in os.walk(proj_dir):  # type: str, List[str], List[str]
        for filename in filter(lambda f: os.path.splitext(f)[1] == '.csv',
                               files):  # type: str
            # pylint: disable=C0103
            with open(os.path.join(root, filename)) as f:  # type: TextIO
                count += len(f.readlines())
    return count


def get_projects(data_dir: str) -> List[Tuple[str, int]]:
    """Get projects from directory with number of type annotations in each"""
    return [(proj, count_annots(os.path.join(data_dir, proj)))
            for proj in os.listdir(data_dir)]


def read_splits(filename: str) -> Dict[str, Tuple[float, float, float]]:
    """Read specification of groups from file"""
    result: Dict[str, Tuple[float, float, float]] = {}
    # pylint: disable=C0103
    with open(filename) as f:  # type: TextIO
        for line in f:  # type: str
            name, min_str, goal_str, max_str\
                = line.split(',')  # type: str, str, str, str
            result[name] = (float(min_str), float(goal_str), float(max_str))
    return result


def write_project(outfile: TextIO, proj_dir: str) -> None:
    """TODO"""
    for root, _, files in os.walk(proj_dir):  # type: str, List[str], List[str]
        for filename in filter(lambda f: os.path.splitext(f)[1] == '.csv',
                               files):  # type: str
            # pylint: disable=C0103
            with open(os.path.join(root, filename)) as f:  # type: TextIO
                outfile.writelines(f.readlines())


def write_split(outfilename: str, projdir: str, projs: List[str]) -> None:
    """TODO"""
    myutil.ensure_parents(outfilename)
    with open(outfilename, 'w', newline='') as outfile:  # type: TextIO
        for project in projs:  # type: str
            write_project(outfile, os.path.join(projdir, project))


if __name__ == "__main__":
    PARSER: ArgumentParser = ArgumentParser(
        description='Split projects into groups')
    PARSER.add_argument('datadir',
                        help='directory with projects')
    PARSER.add_argument('splits',
                        help='file specifying group sizes as'
                        + '(name, min_fraction, max_fraction)')  # noqa
    PARSER.add_argument('outdir', nargs='?',
                        help='directory to save output files in')  # noqa
    ARGS: Namespace = PARSER.parse_args()

    splits: Optional[Mapping[str, Tuple[float, List[str]]]]\
        = get_split(read_splits(ARGS.splits), get_projects(ARGS.datadir))

    if splits is not None:
        for split, (fraction, project_list) in splits.items():\
                # type: str, Tuple[float, List[str]]
            out_filename: str\
                = os.path.join(ARGS.outdir
                               if ARGS.outdir is not None
                               else ARGS.datadir,
                               split + '.csv')
            write_split(out_filename, ARGS.datadir, project_list)
            print(split + ': ' + str(fraction))
