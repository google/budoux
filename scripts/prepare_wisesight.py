# Copyright 2023 Google LLC
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
"""Prepares a dataset from the Wisesight corpus.

Before running this script, download the Wisesight corpus by running:

$ curl -o wisesight-1000-samples-tokenised.label https://raw.githubusercontent.com/PyThaiNLP/wisesight-sentiment/master/word-tokenization/wisesight-1000-samples-tokenised.label

Then run this command to generate the full dataset and optional 3-way splits:

$ python scripts/prepare_wisesight.py wisesight-1000-samples-tokenised.label -o source_wisesight.txt --split-dir tmp/splits
"""

import argparse
import os
import random
import re
import sys

import regex

# module hack
LIB_PATH = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(LIB_PATH))

from budoux import utils

DEFAULT_OUT_PATH = 'source.txt'
EMOJI_PATTERN = regex.compile(
  r'[\p{Emoji_Presentation=Yes}\p{Extended_Pictographic}\p{Emoji_Modifier}\ufe0e\ufe0f]+\p{M}*'
)


def preprocess_line(line: str) -> str:
  """Cleans a raw Wisesight corpus line and converts pipe separators to SEP.

  Removes invisible format characters (such as zero-width spaces), URLs,
  hashtags, and emojis (along with variation selectors, skin-tone modifiers, and
  attached combining marks), collapses consecutive separators and redundant
  space chunks, and strips leading/trailing separators.

  Args:
    line: Raw pipe-segmented line from the Wisesight label file.

  Returns:
    Cleaned BudouX-segmented line using `utils.SEP` (`▁`).
  """
  line = regex.compile(r'\p{Cf}').sub('', line.strip())
  line = re.sub(r'https?://[^ ]+', '', line)  # Remove URLs
  line = re.sub(r'#[^ ]+', '', line)  # Remove hashtags
  line = EMOJI_PATTERN.sub('', line)  # Remove emojis and modifiers
  line = re.sub(r'\|+', '|', line)  # Remove consecutive separators
  line = re.sub(r'\|(\s+\|)+', '| |', line)  # Collapse consecutive space tokens
  line = re.sub(r'^(\s*\|)+', '', line)  # Remove leading separators and spaces
  line = re.sub(r'(\|\s*)+$', '', line)  # Remove trailing separators and spaces
  line = line.strip()
  return line.replace('|', utils.SEP)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawTextHelpFormatter
  )
  parser.add_argument('source_filepath', help='Path to a Wisesight corpus label file.')
  parser.add_argument(
    '-o',
    '--outfile',
    help=f'File path to the output dataset. (default: {DEFAULT_OUT_PATH})',
    default=DEFAULT_OUT_PATH,
  )
  parser.add_argument(
    '--split-dir',
    help=(
      'Directory to output 3-way split datasets '
      '(wisesight_train.txt, wisesight_val.txt, wisesight_test.txt).'
    ),
    default=None,
  )
  parser.add_argument(
    '--val-ratio',
    type=float,
    help='Ratio of validation partition (default: 0.10).',
    default=0.10,
  )
  parser.add_argument(
    '--test-ratio',
    type=float,
    help='Ratio of test partition (default: 0.10).',
    default=0.10,
  )
  parser.add_argument(
    '--seed',
    type=int,
    help='Random seed for deterministic split (default: 42).',
    default=42,
  )
  return parser.parse_args(argv)


def process_wisesight(
  source_filepath: str,
  outfile: str = DEFAULT_OUT_PATH,
  split_dir: str | None = None,
  val_ratio: float = 0.10,
  test_ratio: float = 0.10,
  seed: int = 42,
) -> None:
  """Processes a Wisesight label file into segmented text and optional splits."""
  sentences: list[str] = []
  with open(source_filepath, encoding='utf-8') as infile:
    for line in infile:
      cleaned = preprocess_line(line)
      if utils.SEP not in cleaned:
        continue
      sentences.append(cleaned)

  out_dir = os.path.dirname(outfile)
  if out_dir and not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

  with open(outfile, 'w', encoding='utf-8') as f_out:
    f_out.writelines(s + '\n' for s in sentences)
  print(f'\033[92mTraining data is output to: {outfile}\033[0m')

  if split_dir:
    valid_ratios = (
      (0.0 <= val_ratio <= 1.0)
      and (0.0 <= test_ratio <= 1.0)
      and (val_ratio + test_ratio < 1.0)
    )
    if not valid_ratios:
      raise ValueError(
        'Invalid split ratios: val_ratio + test_ratio must be less than 1.0.'
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

    train_path = os.path.join(split_dir, 'wisesight_train.txt')
    val_path = os.path.join(split_dir, 'wisesight_val.txt')
    test_path = os.path.join(split_dir, 'wisesight_test.txt')

    for path, split_lines in [
      (train_path, train_sentences),
      (val_path, val_sentences),
      (test_path, test_sentences),
    ]:
      with open(path, 'w', encoding='utf-8') as f:
        f.writelines(line + '\n' for line in split_lines)

    print(
      f'\033[92m3-Way split dataset written to {split_dir}:\n'
      f'  Train: {train_path} ({len(train_sentences)} lines)\n'
      f'  Val:   {val_path} ({len(val_sentences)} lines)\n'
      f'  Test:  {test_path} ({len(test_sentences)} lines)\033[0m'
    )


def main() -> None:
  args = parse_args()
  process_wisesight(
    source_filepath=args.source_filepath,
    outfile=args.outfile,
    split_dir=args.split_dir,
    val_ratio=args.val_ratio,
    test_ratio=args.test_ratio,
    seed=args.seed,
  )


if __name__ == '__main__':
  main()
