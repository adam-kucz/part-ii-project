#!/usr/bin/env python3
"""Module for downloading python repositories from github"""

from argparse import ArgumentParser, Namespace
from os import chdir, walk
from pathlib import Path
import re
from shutil import rmtree
from typing import Iterable, List, Set, Tuple
from subprocess import CompletedProcess, DEVNULL, run, PIPE  # nosec

from process_python_input import extract_type_annotations


def download_data(repos: Iterable[str],
                  ignore_existing: bool = False) -> List[str]:
    """Attempts to find and download given repositories"""
    not_found: List[str] = []
    for reponame in repos:  # type: str
        repo_path = Path(reponame)
        if repo_path.exists():
            if ignore_existing:
                rmtree(reponame, ignore_errors=True)
            else:
                continue
        result: CompletedProcess\
            = run(["repogit", reponame],
                  stdout=PIPE, stderr=DEVNULL,
                  universal_newlines=True)  # nosec
        match = re.search("([^/]*)/" + reponame + "]", result.stdout)
        if match is None:
            print("Repository '" + reponame + "' not found")
            not_found.append(reponame)
            continue
        username: str = match.group(1)
        run(["git", "clone",  # nosec
             "git@github.com:" + username + "/" + reponame + ".git"])
    return not_found


def remove_non_python(data_dir: str) -> None:
    """Delete all non-python files and resulting empty directories"""
    for root, dirs, files in walk(data_dir, topdown=False):\
            # type: str, List[str], List[str]
        root_path: Path = Path(root)
        for name in files:  # type: str
            if Path(name).suffix != ".py":
                root_path.joinpath(name).unlink()
        for name in dirs:  # type: str
            try:
                root_path.joinpath(name).rmdir()
            except OSError:
                continue


def extract_annotations(repo_dir: Path,
                        out_dir: Path,
                        fun_as_ret: bool = False)\
                        -> List[Tuple[str, Exception]]:
    """
    Extracts annotaions from all files in the directory

    Stores the files in per-repo subdirectories of out_dir
    """
    exceptions: List[Path, Exception] = []
    for pypath in filter(lambda p: p.is_file(), repo_dir.rglob('*.py')):\
            # type: Path
        rel: Path = pypath.relative_to(repo_dir)
        repo: str = rel.parts[0]
        outpath: Path = out_dir.joinpath(repo, '+'.join(rel.parts[1:]))\
                               .with_suffix('.csv')
        try:
            extract_type_annotations(pypath, outpath, fun_as_ret)
        except (SyntaxError, UnicodeDecodeError) as exception:
            exceptions.append((pypath, exception))
    return exceptions


if __name__ == "__main__":
    PARSER: ArgumentParser = ArgumentParser(
        description='Download python repositories from github')
    PARSER.add_argument('path', type=Path,
                        help='source file, with one repository per line')
    PARSER.add_argument('repodir', type=Path, nargs='?', default=Path('repos'),
                        help='directory with all repositories')
    PARSER.add_argument('outdir', type=Path, nargs='?', default=Path('data'),
                        help='output directory for extracted types')
    PARSER.add_argument('-r', action='store_true',
                        help='redownload repositories even if already exist')
    PARSER.add_argument('-f', action='store_true',
                        help='save functions as their return types')
    ARGS: Namespace = PARSER.parse_args()

    with ARGS.path.open() as infile:  # type: TextIO
        matches: Iterable = (re.search("^\\./(.*?)/", line)
                             for line in infile)
        REPOS: Set[str] = set(match.group(1) for match in matches
                              if match is not None)

    if not ARGS.repodir.exists():
        ARGS.repodir.mkdir(parents=True)
    if not ARGS.outdir.exists():
        ARGS.outdir.mkdir(parents=True)
    cwd: Path = Path.cwd()
    chdir(ARGS.repodir)
    NOT_FOUND: List[str] = download_data(REPOS, ARGS.r)
    chdir(cwd)

    if NOT_FOUND:
        print("Missing {} repositories: {}".format(len(NOT_FOUND), NOT_FOUND))
    else:
        print("All {} repositories downloaded".format(len(REPOS)))

    print("Removing non-python files from {}".format(ARGS.repodir))
    remove_non_python(ARGS.repodir)
    print("Extracting types from {}".format(ARGS.outdir))
    extract_annotations(ARGS.repodir, ARGS.outdir, ARGS.f)
