#!/usr/bin/env python3
"""Module for downloading python repositories from github"""

from argparse import ArgumentParser, Namespace
import os
import re
from typing import Iterable, List, Set
from subprocess import CompletedProcess, DEVNULL, run, PIPE  # nosec

if __name__ == "__main__":
    PARSER: ArgumentParser = ArgumentParser(
        description='Download python repositories from github')
    PARSER.add_argument('path',
                        help='source file, with one repository per line')
    PARSER.add_argument('outdir', nargs='?', default='data',
                        help='output directory for all repositories')
    ARGS: Namespace = PARSER.parse_args()

    with open(ARGS.path, 'r') as infile:  # type: TextIO
        matches: Iterable = (re.search("^\\./(.*?)/", line)
                             for line in infile)
        repos: Set[str] = set(match.group(1) for match in matches
                              if match is not None)

    if not os.path.exists(ARGS.outdir):
        os.makedirs(ARGS.outdir)
    os.chdir(ARGS.outdir)

    not_found: List[str] = []
    for reponame in repos:
        if os.path.exists(reponame):
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
    if not_found:
        print("Missing " + str(len(not_found)) + " repositories: "
              + str(not_found))  # noqa: W503
    else:
        print("All " + str(len(repos)) + " repositories downloaded")
