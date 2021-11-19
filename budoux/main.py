#!/usr/bin/env python
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
    if os.path.isfile(path):
        return path
    else:
        raise FileNotFoundError("'{}' is not found.".format(path))


def parse_args(test: typing.Optional[typing.List[str]] = None) -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        prog="budoux",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
          BudouX is the successor to Budou,
          the machine learning powered line break organizer tool."""
        ),
    )

    parser.add_argument("text", metavar="TXT", type=str, help="text")
    parser.add_argument(
        "-H",
        "--html",
        action="store_true",
        help="HTML mode",
    )
    parser.add_argument(
        "-s", "--stdin", action="store_true", help="Read input from stdin"
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

    if args.stdin:
        inputs = sys.stdin.read()
    else:
        inputs = args.text

    if args.html:
        print(parser.translate_html_string(inputs))
    else:
        print(parser.parse(inputs))


if __name__ == "__main__":
    main()
