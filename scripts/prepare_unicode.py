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
import random
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "lang", help="Language or model identifier (e.g. 'ja', 'ja_word', 'th', 'ko')."
  )
  parser.add_argument(
    "-o",
    "--outfile",
    help="File path to the output dataset. (default: data/unicode_<lang>.txt)",
    default=None,
  )
  parser.add_argument(
    "--split-dir",
    help=(
      "Directory to output 3-way split datasets "
      "(unicode_<lang>_train.txt, unicode_<lang>_val.txt, unicode_<lang>_test.txt)."
    ),
    default=None,
  )
  parser.add_argument(
    "--val-ratio",
    type=float,
    help="Ratio of validation partition (default: 0.10).",
    default=0.10,
  )
  parser.add_argument(
    "--test-ratio",
    type=float,
    help="Ratio of test partition (default: 0.10).",
    default=0.10,
  )
  parser.add_argument(
    "--seed",
    type=int,
    help="Random seed for deterministic split (default: 42).",
    default=42,
  )
  return parser.parse_args(argv)


def process_unicode(
  lang: str,
  outfile: str | None = None,
  split_dir: str | None = None,
  val_ratio: float = 0.10,
  test_ratio: float = 0.10,
  seed: int = 42,
  corpus_dir: str = TEMP_DIR,
) -> None:
  """Processes Unicode test-corpora files into segmented text and optional splits."""
  outfile = outfile or f"data/unicode_{lang}.txt"
  base_lang, _, _ = lang.partition("_")

  out_dir = os.path.dirname(outfile)
  if out_dir and not os.path.exists(out_dir):
    os.makedirs(out_dir)

  if not os.path.exists(corpus_dir):
    print(f"Downloading {REPO_URL} to {corpus_dir}...")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, corpus_dir], check=True)

  pattern = os.path.join(
    corpus_dir, "gutenberg", "*", "out", "google", "breakpoints", base_lang, "*.txt"
  )
  files = sorted(glob.glob(pattern))
  files = [f for f in files if "wrap" not in f]
  if not files:
    print(
      f"No breakpoint files found for language '{lang}' (directory '{base_lang}') "
      f"in {corpus_dir}.",
      file=sys.stderr,
    )
    sys.exit(1)

  print(f"Found {len(files)} {base_lang} breakpoint files in {corpus_dir}.")

  sentences: list[str] = []
  for f_in_path in files:
    with open(f_in_path, encoding="utf-8") as f_in:
      for line in f_in:
        line = line.strip()
        if not line:
          continue
        converted = convert_line(line)
        if converted:
          sentences.append(converted)

  with open(outfile, "w", encoding="utf-8") as f_out:
    f_out.writelines(s + "\n" for s in sentences)

  print(f"Successfully wrote {len(sentences)} lines to {outfile}.")

  if split_dir:
    valid_ratios = (
      (0.0 <= val_ratio <= 1.0)
      and (0.0 <= test_ratio <= 1.0)
      and (val_ratio + test_ratio < 1.0)
    )
    if not valid_ratios:
      raise ValueError(
        "Invalid split ratios: val_ratio + test_ratio must be less than 1.0."
      )
    os.makedirs(split_dir, exist_ok=True)
    shuffled = sentences.copy()
    random.seed(seed)
    random.shuffle(shuffled)

    num_total = len(shuffled)
    num_val = int(num_total * val_ratio)
    num_test = int(num_total * test_ratio)
    num_train = num_total - num_val - num_test

    train_sentences = shuffled[:num_train]
    val_sentences = shuffled[num_train : num_train + num_val]
    test_sentences = shuffled[num_train + num_val :]

    train_path = os.path.join(split_dir, f"unicode_{lang}_train.txt")
    val_path = os.path.join(split_dir, f"unicode_{lang}_val.txt")
    test_path = os.path.join(split_dir, f"unicode_{lang}_test.txt")

    for path, split_lines in [
      (train_path, train_sentences),
      (val_path, val_sentences),
      (test_path, test_sentences),
    ]:
      with open(path, "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in split_lines)

    print(
      f"\033[92m3-Way split dataset written to {split_dir}:\n"
      f"  Train: {train_path} ({len(train_sentences)} lines)\n"
      f"  Val:   {val_path} ({len(val_sentences)} lines)\n"
      f"  Test:  {test_path} ({len(test_sentences)} lines)\033[0m"
    )


def main() -> None:
  args = parse_args()
  process_unicode(
    lang=args.lang,
    outfile=args.outfile,
    split_dir=args.split_dir,
    val_ratio=args.val_ratio,
    test_ratio=args.test_ratio,
    seed=args.seed,
  )


if __name__ == "__main__":
  main()
