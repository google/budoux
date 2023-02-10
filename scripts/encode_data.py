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
"""Encodes the training data with extracted features."""

import argparse
import functools
import itertools
import multiprocessing
import sys
import typing

from budoux import utils

ArgList = typing.Optional[typing.List[str]]
DEFAULT_OUTPUT_FILENAME = 'encoded_data.txt'

INVALID = '▔'
"""The invalid feature string."""


def get_feature(w1: str, w2: str, w3: str, w4: str, w5: str,
                w6: str) -> typing.List[str]:
  """Generates a feature from characters around (w1-6).

  Args:
    w1 (str): The character 3 characters before the break point.
    w2 (str): The character 2 characters before the break point.
    w3 (str): The character right before the break point.
    w4 (str): The character right after the break point.
    w5 (str): The character 2 characters after the break point.
    w6 (str): The character 3 characters after the break point.

  Returns:
    The feature (list[str]).

  """
  raw_feature = {
      'UW1': w1,
      'UW2': w2,
      'UW3': w3,
      'UW4': w4,
      'UW5': w5,
      'UW6': w6,
      'BW1': w2 + w3,
      'BW2': w3 + w4,
      'BW3': w4 + w5,
      'TW1': w1 + w2 + w3,
      'TW2': w2 + w3 + w4,
      'TW3': w3 + w4 + w5,
      'TW4': w4 + w5 + w6,
  }
  for key, value in list(raw_feature.items()):
    if INVALID in value:
      del raw_feature[key]
  return [f'{item[0]}:{item[1]}' for item in raw_feature.items()]


def parse_args(test: ArgList = None) -> argparse.Namespace:
  """Parses commandline arguments.

  Args:
    test (typing.Optional[typing.List[str]], optional): Commandline args for testing. Defaults to None.

  Returns:
    argparse.Namespace: Parsed data of args.
  """
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      'source_data',
      help='''File path of the source training data to extract features.''')
  parser.add_argument(
      '-o',
      '--outfile',
      help='''Output file path for the encoded training data.
            (default: encoded_data.txt)''',
      default=DEFAULT_OUTPUT_FILENAME)
  parser.add_argument(
      '--processes',
      type=int,
      help='''Number of processes to use.
          (default: the number of CPUs in the system)''',
      default=None)
  if test is None:
    return parser.parse_args()
  else:
    return parser.parse_args(test)


def process(i: int, sentence: str, sep_indices: typing.Set[int]) -> str:
  """Outputs an encoded line of features from the given index.

  Args:
    i (int): index
    sentence (str): A sentence
    sep_indices (typing.Set[int]): A set of separator indices.
  """
  feature = get_feature(sentence[i - 3] if i > 2 else INVALID,
                        sentence[i - 2] if i > 1 else INVALID, sentence[i - 1],
                        sentence[i] if i < len(sentence) else INVALID,
                        sentence[i + 1] if i + 1 < len(sentence) else INVALID,
                        sentence[i + 2] if i + 2 < len(sentence) else INVALID)
  positive = i in sep_indices
  line = '\t'.join(['1' if positive else '-1'] + feature)
  return line


def normalize_input(data: str) -> typing.Tuple[str, typing.Set[int]]:
  """Normalizes the input to one line with separators.

  Args:
    data(str): Source input

  Returns:
    typing.Tuple[str, typing.Set[int]]: A tuple of the sentence and the
      separator indices.
  """
  chunks = data.replace('\n', utils.SEP).strip().split(utils.SEP)
  chunk_lengths = [len(chunk) for chunk in chunks]
  sep_indices = set(itertools.accumulate(chunk_lengths, lambda x, y: x + y))
  sentence = ''.join(chunks)
  return (sentence, sep_indices)


def main(test: ArgList = None) -> None:
  args = parse_args(test)
  source_filename: str = args.source_data
  entries_filename: str = args.outfile
  processes = None if args.processes is None else int(args.processes)
  with open(source_filename, encoding=sys.getdefaultencoding()) as f:
    data = f.read()
  sentence, sep_indices = normalize_input(data)
  with multiprocessing.Pool(processes) as p:
    func = functools.partial(
        process, sentence=sentence, sep_indices=sep_indices)
    lines = p.map(func, range(1, len(sentence) + 1))

  with open(entries_filename, 'w', encoding=sys.getdefaultencoding()) as f:
    for line in lines:
      f.write(line + '\n')

  print('\033[92mEncoded training data is out at: %s\033[0m' % entries_filename)


if __name__ == '__main__':
  main()
