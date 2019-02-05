#!/usr/bin/env python3
"""Module for downloading python repositories from github"""

from argparse import ArgumentParser, Namespace
import os
import re
import shutil
from typing import Iterable, List, Set, Tuple
from subprocess import CompletedProcess, DEVNULL, run, PIPE  # nosec

from process_python_input import extract_type_annotations
from util import ensure_dir


def download_data(repos: Iterable[str],
                  ignore_existing: bool = False) -> List[str]:
    """Attempts to find and download given repositories"""
    not_found: List[str] = []
    for reponame in repos:  # type: str
        if os.path.exists(reponame):
            if ignore_existing:
                shutil.rmtree(reponame, ignore_errors=True)
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
    for root, dirs, files in os.walk(data_dir, topdown=False):\
            # type: str, List[str], List[str]
        for name in files:  # type: str
            _, ext = os.path.splitext(name)  # type: str, str
            if ext != ".py":
                os.remove(os.path.join(root, name))
        for name in dirs:  # type: str
            try:
                os.rmdir(os.path.join(root, name))
            except OSError:
                continue


def extract_annotations(data_dir: str) -> List[Tuple[str, Exception]]:
    """Extracts annotaions from all files in the directory"""
    exceptions: List[str, Exception] = []
    for root, _, files in os.walk(data_dir):  # type: str, List[str], List[str]
        for name in files:  # type: str
            base, ext = os.path.splitext(name)  # type: str, str
            if ext == ".py":
                abs_name = os.path.join(root, name)
                try:
                    extract_type_annotations(abs_name,
                                             os.path.join(root, base + '.csv'))
                except (SyntaxError, UnicodeDecodeError) as exception:
                    exceptions.append((abs_name, exception))
    return exceptions


if __name__ == "__main__":
    PARSER: ArgumentParser = ArgumentParser(
        description='Download python repositories from github')
    PARSER.add_argument('path',
                        help='source file, with one repository per line')
    PARSER.add_argument('outdir', nargs='?', default='data',
                        help='output directory for all repositories')
    PARSER.add_argument('-r', action='store_true',
                        help='redownload repositories even if already exist')
    ARGS: Namespace = PARSER.parse_args()

    with open(ARGS.path, 'r') as infile:  # type: TextIO
        matches: Iterable = (re.search("^\\./(.*?)/", line)
                             for line in infile)
        REPOS: Set[str] = set(match.group(1) for match in matches
                              if match is not None)

    ensure_dir(ARGS.outdir)
    cwd: str = os.getcwd()
    os.chdir(ARGS.outdir)
    NOT_FOUND: List[str] = download_data(REPOS, ARGS.r)
    os.chdir(cwd)

    if NOT_FOUND:
        print("Missing " + str(len(NOT_FOUND)) + " repositories: "
              + str(NOT_FOUND))  # noqa: W503
    else:
        print("All " + str(len(REPOS)) + " repositories downloaded")

    print("Removing non-python files from " + ARGS.outdir)
    remove_non_python(ARGS.outdir)
    print("Extracting types from " + ARGS.outdir)
    extract_annotations(ARGS.outdir)
