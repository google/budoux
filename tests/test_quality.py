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
"""Quality regression test."""

import os
import sys
import unittest

# module hack
LIB_PATH = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(LIB_PATH))

from budoux import load_default_japanese_parser, utils  # noqa (module hack)


class TestQuality(unittest.TestCase):

  def test_ja(self) -> None:
    parser = load_default_japanese_parser()
    fp = os.path.join(os.path.dirname(__file__), 'quality', 'ja.tsv')
    with open(fp, 'r', encoding='utf-8') as f:
      data = [line.split('\t') for line in f.readlines() if line[0] != '#']
    expected_sentences = [line[1].strip() for line in data if len(line) > 1]
    for expected in expected_sentences:
      result = utils.SEP.join(parser.parse(expected.replace(utils.SEP, '')))
      self.assertEqual(result, expected)
