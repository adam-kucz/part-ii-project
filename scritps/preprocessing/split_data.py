#!/usr/bin/env python3
"""Module for splitting data into subsets"""

from argparse import ArgumentParser, Namespace
from ast import literal_eval
from itertools import dropwhile
from os import walk
from pathlib import Path
from queue import PriorityQueue
import random
from typing import (Any, Dict, Iterable, List, Mapping,  # noqa: F401
                    Optional, Sequence, TextIO, Tuple, TypeVar)


T = TypeVar('T')  # pylint: disable=invalid-name


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
    # pylint: disable=unsubscriptable-object
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
        if found:
            proj, num = found  # type: str, float
            result[group] = (g_size + num, g_list + [proj])
            projects_normalised.remove(found)
            queue.put_nowait((abs(result[group][0] - groups[group][1]), group))
    return result if not projects_normalised else None


# TODO: fix
# pylint: disable=too-many-locals
def get_split(
        requested: Mapping[T, Tuple[float, float, float]],
        projects_iter: Iterable[Tuple[str, int]]) -> \
            Optional[Dict[T, Tuple[float, List[str]]]]:  # noqa: E126
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
            if found:
                proj, num = found  # type: str, float
                group_size += num
                group_projs.append(proj)
                projects_normalised.remove(found)
            else:
                return None
        result[subset] = (group_size, group_projs)
    return with_added_remaining(result, projects_normalised, requested)


def count_annots(proj_dir: Path) -> int:
    """Count number of identifier/type pairs in project"""
    count = 0
    for root, _, files in walk(proj_dir):  # type: str, List[str], List[str]
        for filename in filter(lambda f: f.suffix == '.csv',
                               map(Path, files)):  # type: Path
            # pylint: disable=invalid-name
            with Path(root).joinpath(filename).open() as f:  # type: TextIO
                count += len(f.readlines())
    return count


def get_projects(data_dir: Path) -> List[Tuple[Path, int]]:
    """Get projects from directory with number of type annotations in each"""
    return [(proj, count_annots(data_dir.joinpath(proj)))
            for proj in data_dir.iter_dir() if proj.is_dir()]


def read_splits(filename: Path) -> Dict[str, Tuple[float, float, float]]:
    """Read specification of groups from file"""
    result: Dict[str, Tuple[float, float, float]] = {}
    # pylint: disable=invalid-name
    with filename.open() as f:  # type: TextIO
        for line in f:  # type: str
            name, min_str, goal_str, max_str\
                = line.split(',')  # type: str, str, str, str
            result[name] = (float(min_str), float(goal_str), float(max_str))
    return result


def write_project(outfile: TextIO, proj_dir: Path) -> None:
    """TODO"""
    for filepath in proj_dir.rglob("*.csv"):
        if filepath.is_file():
            with filepath.open() as csvfile:  # type: TextIO
                outfile.writelines(csvfile.readlines())


def write_split(outfilename: Path, projdir: Path, projects: List[str]) -> None:
    """TODO"""
    if not outfilename.parent.exists():
        outfilename.parent.mkdir(parents=True)
    with outfilename.open('w', newline='') as outfile:  # type: TextIO
        for project in projects:  # type: str
            write_project(outfile, projdir.joinpath(project))


if __name__ == "__main__":
    PARSER: ArgumentParser = ArgumentParser(
        description='Split projects into groups')
    PARSER.add_argument('datadir', type=Path,
                        help='directory with projects')
    PARSER.add_argument('splits', type=Path,
                        help='file specifying group sizes as'
                        + '(name, min_fraction, max_fraction)')  # noqa
    PARSER.add_argument('outdir', nargs='?', type=Path,
                        help='directory to save output files in')
    PARSER.add_argument('--log_file', default=Path('log.txt'), type=Path,
                        help='file to save splits in')  # noqa
    PARSER.add_argument('-r', action='store_true',
                        help='read splits from log file')  # noqa
    ARGS: Namespace = PARSER.parse_args()

    # pylint: disable=invalid-name
    outdir = ARGS.outdir if ARGS.outdir else ARGS.datadir

    if ARGS.r:
        with ARGS.log_file.open() as log:
            for split_line in log:
                split: str = split_line.split()[0]
                projs: List[str] = literal_eval(split_line.split(': ')[1])
                out_filename: Path = outdir.joinpath(split).with_suffix('.csv')
                write_split(out_filename, ARGS.datadir, projs)
    else:
        splits: Optional[Mapping[str, Tuple[float, List[str]]]]\
            = get_split(read_splits(ARGS.splits), get_projects(ARGS.datadir))
        if splits:
            with ARGS.log_file.open('w') as log:
                for split, (fraction, project_list) in splits.items():\
                        # type: str, Tuple[float, List[str]]
                    print("{} ({}%): {}"
                          .format(split, fraction * 100, project_list),
                          file=log)
                    out_filename: Path = outdir.joinpath(split)\
                                               .with_suffix('.csv')
                    write_split(out_filename, ARGS.datadir, project_list)
                    print("{}: {}".format(split, fraction))
