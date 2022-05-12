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
import json
import os
import typing

from .utils import INVALID

with open(os.path.join(os.path.dirname(__file__), 'unicode_blocks.json')) as f:
  block_starts: typing.List[int] = json.load(f)


def unicode_block_index(w: str) -> str:
  """Returns the index of the Unicode block that the character belongs to.

  Args:
    w (str): A character.

  Returns:
    index (str): Unicode block index in three digits.
  """
  if not w or w == INVALID:
    return INVALID
  return '%03d' % (bisect.bisect_right(block_starts, ord(w[0])))


def get_feature(w1: str, w2: str, w3: str, w4: str, w5: str, w6: str, p1: str,
                p2: str, p3: str) -> typing.List[str]:
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
  b1 = unicode_block_index(w1)
  b2 = unicode_block_index(w2)
  b3 = unicode_block_index(w3)
  b4 = unicode_block_index(w4)
  b5 = unicode_block_index(w5)
  b6 = unicode_block_index(w6)
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
  for key, value in list(raw_feature.items()):
    if INVALID in value:
      del raw_feature[key]
  return [f'{item[0]}:{item[1]}' for item in raw_feature.items()]
