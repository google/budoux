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
"""Methods to encode source sentences to features."""

import bisect
import itertools
import json
import os
import sys
import typing
from .utils import SEP, Result

with open(os.path.join(os.path.dirname(__file__), 'unicode_blocks.json')) as f:
  block_starts: typing.List[int] = json.load(f)


def unicode_block_index(w: str):
  """Returns the index of the Unicode block that the character belongs to.

  Args:
    w (str): A character.

  Returns:
    index (int): Unicode block index.
  """
  return bisect.bisect_right(block_starts, ord(w))


def get_feature(w1: str, w2: str, w3: str, w4: str, w5: str, w6: str, p1: str,
                p2: str, p3: str):
  """Generates a feature from characters around (w1-6) and past results (p1-3).

  Args:
    w1 (str): The character 3 characters before the break point.
    w2 (str): The character 2 characters before the break point.
    w3 (str): The character right before the break point.
    w4 (str): The character right after the break point.
    w5 (str): The character 2 characters after the break point.
    w6 (str): The character 3 characters after the break point.
    p1 (str): The result 3 steps ago.
    p2 (str): The result 2 steps ago.
    p3 (str): The last result.

  Returns:
    The feature (list[str]).

  """
  b1 = '%03d' % (unicode_block_index(w1)) if w1 != '' else '999'
  b2 = '%03d' % (unicode_block_index(w2)) if w2 != '' else '999'
  b3 = '%03d' % (unicode_block_index(w3)) if w3 != '' else '999'
  b4 = '%03d' % (unicode_block_index(w4)) if w4 != '' else '999'
  b5 = '%03d' % (unicode_block_index(w5)) if w5 != '' else '999'
  b6 = '%03d' % (unicode_block_index(w6)) if w6 != '' else '999'
  raw_feature = {
      'UP1': p1,
      'UP2': p2,
      'UP3': p3,
      'BP1': p1 + p2,
      'BP2': p2 + p3,
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
      'UB1': b1,
      'UB2': b2,
      'UB3': b3,
      'UB4': b4,
      'UB5': b5,
      'UB6': b6,
      'BB1': b2 + b3,
      'BB2': b3 + b4,
      'BB3': b4 + b5,
      'TB1': b1 + b2 + b3,
      'TB2': b2 + b3 + b4,
      'TB3': b3 + b4 + b5,
      'TB4': b4 + b5 + b6,
      'UQ1': p1 + b1,
      'UQ2': p2 + b2,
      'UQ3': p3 + b3,
      'BQ1': p2 + b2 + b3,
      'BQ2': p2 + b3 + b4,
      'BQ3': p3 + b2 + b3,
      'BQ4': p3 + b3 + b4,
      'TQ1': p2 + b1 + b2 + b3,
      'TQ2': p2 + b2 + b3 + b4,
      'TQ3': p3 + b1 + b2 + b3,
      'TQ4': p3 + b2 + b3 + b4,
  }
  if raw_feature['UW4'] == '':
    del raw_feature['UW4']
  if raw_feature['UW5'] == '':
    del raw_feature['UW5']
  if raw_feature['UW6'] == '':
    del raw_feature['UW6']
  if raw_feature['BW3'] == '':
    del raw_feature['BW3']
  if raw_feature['TW4'] == '':
    del raw_feature['TW4']
  if raw_feature['UB4'] == '999':
    del raw_feature['UB4']
  if raw_feature['UB5'] == '999':
    del raw_feature['UB5']
  if raw_feature['UB6'] == '999':
    del raw_feature['UB6']
  if raw_feature['BB3'] == '999999':
    del raw_feature['BB3']
  if raw_feature['TB4'] == '999999999':
    del raw_feature['TB4']
  return [f'{item[0]}:{item[1]}' for item in raw_feature.items()]


def process(source_filename: str, entries_filename: str):
  """Extratcs features from source sentences and outputs as entries.

  Args:
    source_filename (str): A file path to the source sentences.
    entries_filename (str): A file path to the output entries.
  """
  with open(source_filename, encoding=sys.getdefaultencoding()) as f:
    data = f.readlines()
  with open(entries_filename, 'w', encoding=sys.getdefaultencoding()) as f:
    f.write('')

  for row in data:
    chunks = row.strip().split(SEP)
    chunk_lengths = [len(chunk) for chunk in chunks]
    sep_indices = set(itertools.accumulate(chunk_lengths, lambda x, y: x + y))
    sentence = ''.join(chunks)
    p1 = Result.UNKNOWN.value
    p2 = Result.UNKNOWN.value
    p3 = Result.UNKNOWN.value
    for i in range(3, len(sentence) + 1):
      feature = get_feature(sentence[i - 3], sentence[i - 2], sentence[i - 1],
                            sentence[i] if i < len(sentence) else '',
                            sentence[i + 1] if i + 1 < len(sentence) else '',
                            sentence[i + 2] if i + 2 < len(sentence) else '',
                            p1, p2, p3)
      positive = i in sep_indices
      p = Result.POSITIVE.value if positive else Result.NEGATIVE.value
      with open(entries_filename, 'a', encoding=sys.getdefaultencoding()) as f:
        row = ['1' if positive else '-1'] + feature
        f.write('\t'.join(row) + '\n')
      p1 = p2
      p2 = p3
      p3 = p
