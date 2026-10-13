# Copyright 2026 Google LLC
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
"""Prepares a dataset from the Unicode test-corpora for a given language."""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import tempfile

# module hack
LIB_PATH = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(LIB_PATH))

from budoux import utils

REPO_URL = "https://github.com/unicode-org/test-corpora.git"
TEMP_DIR = os.path.join(tempfile.gettempdir(), "test-corpora")


def convert_line(line: str) -> str:
  """Converts a pipe-segmented line into a BudouX-segmented line.

  Args:
    line: Pipe-segmented line (e.g. 'アリス|は|、|姉|の')

  Returns:
    BudouX-segmented line (e.g. 'アリス▁は▁、▁姉▁の')
  """
  parts = line.strip().split("|")
  chunks = [part for part in parts if part != ""]
  return utils.SEP.join(chunks)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("lang", help="Language code (e.g. 'ja', 'zh', 'th', 'ko').")
  parser.add_argument(
    "-o",
    "--outfile",
    help="File path to the output dataset. (default: data/unicode_<lang>.txt)",
    default=None,
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  lang = args.lang
  outfile = args.outfile or f"data/unicode_{lang}.txt"

  out_dir = os.path.dirname(outfile)
  if out_dir and not os.path.exists(out_dir):
    os.makedirs(out_dir)

  if not os.path.exists(TEMP_DIR):
    print(f"Downloading {REPO_URL} to {TEMP_DIR}...")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, TEMP_DIR], check=True)

  pattern = os.path.join(
    TEMP_DIR, "gutenberg", "*", "out", "google", "breakpoints", lang, "*.txt"
  )
  files = sorted(glob.glob(pattern))
  files = [f for f in files if "wrap" not in f]
  if not files:
    print(
      f"No breakpoint files found for language '{lang}' in {TEMP_DIR}.", file=sys.stderr
    )
    sys.exit(1)

  print(f"Found {len(files)} {lang} breakpoint files in {TEMP_DIR}.")

  total_lines = 0
  with open(outfile, "w", encoding="utf-8") as f_out:
    for f_in_path in files:
      with open(f_in_path, encoding="utf-8") as f_in:
        for line in f_in:
          line = line.strip()
          if not line:
            continue
          converted = convert_line(line)
          if converted:
            f_out.write(converted + "\n")
            total_lines += 1

  print(f"Successfully wrote {total_lines} lines to {outfile}.")


if __name__ == "__main__":
  main()
