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
"""Tests the data encoder script."""

import os
import sys
import unittest
from pathlib import Path

from budoux import utils

# module hack
LIB_PATH = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(LIB_PATH))
from scripts import encode_data  # type: ignore # noqa (module hack)

ENTRIES_FILE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'entries_test.txt'))


class TestEncodeData(unittest.TestCase):

  def setUp(self) -> None:
    Path(ENTRIES_FILE_PATH).touch()

  def test_process(self) -> None:
    separated_sentence = f'これは{utils.SEP}美しい{utils.SEP}ペンです。'
    encode_data.process(separated_sentence, ENTRIES_FILE_PATH)
    with open(
        ENTRIES_FILE_PATH, encoding=sys.getdefaultencoding(),
        errors='replace') as f:
      entries = f.read().splitlines()
    original_sentence = ''.join(separated_sentence.split(utils.SEP))
    self.assertEqual(
        len(entries), len(original_sentence),
        'Should start making entries from the first character.')

    labels = [int(entry.split('\t')[0]) for entry in entries]
    self.assertListEqual(
        labels,
        [
            -1,  # こ
            -1,  # れ
            1,  # は
            -1,  # 美
            -1,  # し
            1,  # い
            -1,  # ペ
            -1,  # ン
            -1,  # で
            -1,  # す
            1  # 。
        ],
        'The first column of entries should be labels.')

    features = [set(entry.split('\t')[1:]) for entry in entries]
    self.assertIn(
        'UW3:こ', features[0],
        'The first feature set should include the first character as the UW3 feature.'
    )
    self.assertIn(
        'UW3:れ', features[1],
        'The second feature set should include the second character as the UW3 feature.'
    )
    self.assertIn(
        'UW3:は', features[2],
        'The third feature set should include the third character as the UW3 feature.'
    )
    self.assertIn(
        'UW3:。', features[-1],
        'The last feature set should include the last character as the UW3 feature.'
    )

  def tearDown(self) -> None:
    os.remove(ENTRIES_FILE_PATH)
