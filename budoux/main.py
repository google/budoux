#!/usr/bin/env python
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BudouX Script to provide CLI for user."""
import argparse
import json
import os
import sys
import textwrap
import typing

import pkg_resources

import budoux

__version__ = "0.0.1"


def check_file(path: str) -> str:
    """Check if filepath is exist or not.

    Args:
        path (str): Model path

    Raises:
        FileNotFoundError: Raise if given path is not exist.

    Returns:
        str: Model path confirmed its existance.
    """
    if os.path.isfile(path):
        return path
    else:
        raise FileNotFoundError("'{}' is not found.".format(path))


def parse_args(test: typing.Optional[typing.List[str]] = None) -> argparse.Namespace:
    """Parse commandline arguments.

    Args:
        test (typing.Optional[typing.List[str]], optional): Commandline args for testing. Defaults to None.

    Returns:
        argparse.Namespace: Parsed data of args.
    """
    parser = argparse.ArgumentParser(
        prog="budoux",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
          BudouX is the successor to Budou,
          the machine learning powered line break organizer tool."""
        ),
    )

    parser.add_argument("text", metavar="TXT", nargs="?", type=str, help="text")
    parser.add_argument(
        "-H",
        "--html",
        action="store_true",
        help="HTML mode",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=check_file,
        default=pkg_resources.resource_filename(__name__, "models/ja-knbc.json"),
    )
    parser.add_argument(
        "-V", "--version", action="version", version="%(prog)s {}".format(__version__)
    )
    if test:
        return parser.parse_args(test)
    else:
        return parser.parse_args()


def main():
    args = parse_args()
    with open(args.model, "r") as f:
        model = json.load(f)

    parser = budoux.Parser(model)

    if args.text is None:
        inputs = sys.stdin.read().rstrip()
    else:
        inputs = args.text

    if args.html:
        print(parser.translate_html_string(inputs))
    else:
        print(parser.parse(inputs))


if __name__ == "__main__":
    main()
