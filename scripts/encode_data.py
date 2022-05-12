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
import itertools
import sys
from budoux import feature_extractor, utils


def process(line: str, entries_filename: str) -> None:
  """Extratcs features from a source sentence and outputs trainig data entries.

  Args:
    source_filename (str): A file path to the source sentences.
    entries_filename (str): A file path to the output entries.
  """
  chunks = line.strip().split(utils.SEP)
  chunk_lengths = [len(chunk) for chunk in chunks]
  sep_indices = set(itertools.accumulate(chunk_lengths, lambda x, y: x + y))
  sentence = ''.join(chunks)
  p1 = utils.Result.UNKNOWN.value
  p2 = utils.Result.UNKNOWN.value
  p3 = utils.Result.UNKNOWN.value
  lines = []
  for i in range(1, len(sentence) + 1):
    feature = feature_extractor.get_feature(
        sentence[i - 3] if i > 2 else utils.INVALID,
        sentence[i - 2] if i > 1 else utils.INVALID, sentence[i - 1],
        sentence[i] if i < len(sentence) else utils.INVALID,
        sentence[i + 1] if i + 1 < len(sentence) else utils.INVALID,
        sentence[i + 2] if i + 2 < len(sentence) else utils.INVALID, p1, p2, p3)
    positive = i in sep_indices
    p = utils.Result.POSITIVE.value if positive else utils.Result.NEGATIVE.value
    lines.append('\t'.join(['1' if positive else '-1'] + feature) + '\n')
    p1 = p2
    p2 = p3
    p3 = p
  with open(entries_filename, 'a', encoding=sys.getdefaultencoding()) as f:
    f.write(''.join(lines))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      'source_data',
      help='''File path of the source training data to extract features.''')
  parser.add_argument(
      '-o',
      '--outfile',
      help='''Output file path for the encoded training data.
            (default: encoded_data.txt)''',
      default='encoded_data.txt')
  args = parser.parse_args()
  source_filename = args.source_data
  entries_filename = args.outfile
  with open(source_filename, encoding=sys.getdefaultencoding()) as f:
    data = f.readlines()
  with open(entries_filename, 'w', encoding=sys.getdefaultencoding()) as f:
    f.write('')
  for line in data:
    process(line, entries_filename)
  print('\033[92mEncoded training data is output to: %s\033[0m' %
        entries_filename)


if __name__ == '__main__':
  main()
