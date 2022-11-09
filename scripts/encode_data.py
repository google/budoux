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
import multiprocessing
import typing
import functools

from budoux import feature_extractor, utils

def process(i: int, sentence: str, sep_indices: typing.Set[int]) -> str:
  feature = feature_extractor.get_feature(
      sentence[i - 3] if i > 2 else utils.INVALID,
      sentence[i - 2] if i > 1 else utils.INVALID, sentence[i - 1],
      sentence[i] if i < len(sentence) else utils.INVALID,
      sentence[i + 1] if i + 1 < len(sentence) else utils.INVALID,
      sentence[i + 2] if i + 2 < len(sentence) else utils.INVALID)
  positive = i in sep_indices
  line = '\t'.join(['1' if positive else '-1'] + feature)
  return line

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
    data = f.read().replace('\n', utils.SEP)
  with open(entries_filename, 'w', encoding=sys.getdefaultencoding()) as f:
    f.write('')
  chunks = data.strip().split(utils.SEP)
  chunk_lengths = [len(chunk) for chunk in chunks]
  sep_indices = set(itertools.accumulate(chunk_lengths, lambda x, y: x + y))
  sentence = ''.join(chunks)

  with multiprocessing.Pool(None) as p:
    lines = p.map(functools.partial(process, sentence=sentence, sep_indices=sep_indices), range(1, len(sentence) + 1))

  with open(entries_filename, 'a', encoding=sys.getdefaultencoding()) as f:
    f.write('\n'.join(lines))

  print('\033[92mEncoded training data is out at: %s\033[0m' % entries_filename)


if __name__ == '__main__':
  main()
