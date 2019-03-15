from ast import literal_eval
from itertools import dropwhile
from os import walk
from pathlib import Path
from queue import PriorityQueue
import random
# noqa justified because mypy needs Any in type comment
from typing import (Any, Dict, IO, Iterable, List, Mapping,  # noqa: F401
                    Optional, Sequence, Tuple, TypeVar)

from parse import parse

__all__ = ["from_logfile", "create_new"]

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
        groups: Mapping[T, Tuple[float, float, float]])\
        -> Optional[Dict[T, Tuple[float, List[str]]]]:
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
            = get_next_project(projects_normalised, groups[group][2] - g_size)
        if found:
            proj, num = found  # type: str, float
            result[group] = (g_size + num, g_list + [proj])
            projects_normalised.remove(found)
            queue.put_nowait((abs(result[group][0] - groups[group][1]), group))
    return result if not projects_normalised else None


def get_split(
        requested: Mapping[T, Tuple[float, float, float]],
        projects_iter: Iterable[Tuple[str, int]])\
        -> Optional[Dict[T, Tuple[float, List[str]]]]:
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
            with Path(root).joinpath(filename).open() as f:  # type: IO
                count += len(f.readlines())
    return count


def get_projects(data_dir: Path) -> List[Tuple[str, int]]:
    """Get projects from directory with number of type annotations in each"""
    return [(str(proj), count_annots(data_dir.joinpath(proj)))
            for proj in data_dir.iterdir() if proj.is_dir()]


def read_splits(filename: Path) -> Dict[str, Tuple[float, float, float]]:
    """Read specification of groups from file"""
    result: Dict[str, Tuple[float, float, float]] = {}
    # pylint: disable=invalid-name
    with filename.open() as f:  # type: IO
        for line in f:  # type: str
            name, min_str, goal_str, max_str\
                = line.split(',')  # type: str, str, str, str
            result[name] = (float(min_str), float(goal_str), float(max_str))
    return result


def write_project(outfile: IO, proj_dir: Path) -> None:
    for filepath in proj_dir.rglob("*.csv"):
        if filepath.is_file():
            outfile.write(filepath.read_text())


def write_split(outfilename: Path, projdir: Path, projects: List[str]) -> None:
    print("Trying to write split {} to file {}"
          .format(projects, outfilename))
    if not outfilename.parent.exists():
        outfilename.parent.mkdir(parents=True)
    with outfilename.open('w', newline='') as outfile:  # type: IO
        for project in projects:  # type: str
            write_project(outfile, projdir.joinpath(project))


def from_logfile(datadir: Path, outdir: Path, logpath: Path):
    for line in logpath.read_text().split('\n'):
        if line:
            lineformat: str = "{split} ({}%): {projs}"
            parsed = parse(lineformat, line)
            if not parsed:
                raise ValueError("Line {} does not conform to format {}"
                                 .format(line, lineformat))
            out_filename: Path = outdir.joinpath(parsed['split'])\
                                       .with_suffix('.csv')
            write_split(out_filename, datadir, literal_eval(parsed['projs']))


def create_new(splitpath: Path, datadir: Path,
               outdir: Path, logpath: Path)\
               -> Optional[Dict[str, Tuple[float, List[str]]]]:
    logfile: IO = logpath.open('w')
    splits: Optional[Mapping[str, Tuple[float, List[str]]]]\
        = get_split(read_splits(splitpath), get_projects(datadir))
    if not splits:
        return None
    results = {}
    for split, (fraction, project_list) in splits.items():\
            # type: str, Tuple[float, List[str]]
        out_filename: Path = outdir.joinpath(split).with_suffix('.csv')
        write_split(out_filename, datadir, project_list)
        results[split] = (fraction * 100, project_list)
        print("{} ({}%): {}".format(split, *results[split]), file=logfile)
    return results
