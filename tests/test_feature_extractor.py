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
"""Tests methods for the feature extractor."""

import io
import os
import sys
import typing
import unittest
from pathlib import Path

# module hack
LIB_PATH = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(LIB_PATH))

from budoux import feature_extractor, utils  # noqa (module hack)

if isinstance(sys.stdin, io.TextIOWrapper) and sys.version_info >= (3, 7):
  sys.stdin.reconfigure(encoding='utf-8')

if isinstance(sys.stdout, io.TextIOWrapper) and sys.version_info >= (3, 7):
  sys.stdout.reconfigure(encoding='utf-8')

SOURCE_FILE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'source_test.txt'))
ENTRIES_FILE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'entries_test.txt'))


class TestFeatureExtractor(unittest.TestCase):

  def setUp(self) -> None:
    Path(ENTRIES_FILE_PATH).touch()
    self.test_entry = f'これは{utils.SEP}美しい{utils.SEP}ペンです。'
    with open(SOURCE_FILE_PATH, 'w', encoding=sys.getdefaultencoding()) as f:
      f.write(self.test_entry)

  def test_unicode_block_index(self) -> None:

    def check(character: str, block: str, msg: str) -> None:
      self.assertEqual(
          feature_extractor.unicode_block_feature(character), block, msg)

    check('a', '001', '"a" should be the 1st block "Basic Latin".')
    check('あ', '108', '"あ" should be the 108th block "Hiragana".')
    check('安', '120', '"安" should be the 120th block "Kanji"')
    check('あ安', '108', 'Only the first character should be recognized')
    check('', utils.INVALID,
          'Should return INVALID when a blank string is given.')
    check(utils.INVALID, utils.INVALID,
          'Should return INVALID when INVALID is given.')

  def test_get_feature(self) -> None:
    feature = feature_extractor.get_feature('a', 'b', 'c', 'd', 'e', 'f', 'x',
                                            'y', 'z')
    self.assertSetEqual(
        set(feature),
        {
            # Unigram of Words (UW)
            'UW1:a',
            'UW2:b',
            'UW3:c',
            'UW4:d',
            'UW5:e',
            'UW6:f',

            # Unigram of Previous Results (UP)
            'UP1:x',
            'UP2:y',
            'UP3:z',

            # Unigram of Unicode Blocks (UB)
            'UB1:001',
            'UB2:001',
            'UB3:001',
            'UB4:001',
            'UB5:001',
            'UB6:001',

            # Combination of UW and UP
            'UQ1:x001',
            'UQ2:y001',
            'UQ3:z001',

            # Bigram of Words (BW), Previous Results (BP), Unicode Blocks (BB), and
            # its combination (BQ)
            'BW1:bc',
            'BW2:cd',
            'BW3:de',
            'BP1:xy',
            'BP2:yz',
            'BB1:001001',
            'BB2:001001',
            'BB3:001001',
            'BQ1:y001001',
            'BQ2:y001001',
            'BQ3:z001001',
            'BQ4:z001001',

            # Trigram of Words (BW), Previous Results (BP), Unicode Blocks (BB), and
            # its combination (BQ)
            'TW1:abc',
            'TW2:bcd',
            'TW3:cde',
            'TW4:def',
            'TB1:001001001',
            'TB2:001001001',
            'TB3:001001001',
            'TB4:001001001',
            'TQ1:y001001001',
            'TQ2:y001001001',
            'TQ3:z001001001',
            'TQ4:z001001001',
        },
        'Features should be extracted.')

    def find_by_prefix(prefix: str, feature: typing.List[str]) -> bool:
      for item in feature:
        if item.startswith(prefix):
          return True
      return False

    feature = feature_extractor.get_feature('a', 'a', utils.INVALID, 'a', 'a',
                                            'a', 'a', 'a', 'a')
    self.assertFalse(
        find_by_prefix('UW3:', feature),
        'Should omit the Unigram feature when the character is invalid.')
    self.assertFalse(
        find_by_prefix('UB3:', feature),
        'Should omit the Unicode block feature when the character is invalid.')
    self.assertFalse(
        find_by_prefix('BW2:', feature),
        'Should omit the Bigram feature that covers an invalid character.')
    self.assertFalse(
        find_by_prefix('BB2:', feature),
        'Should omit the Unicode feature that covers an invalid character.')

  def test_process(self) -> None:
    feature_extractor.process(SOURCE_FILE_PATH, ENTRIES_FILE_PATH)
    with open(
        ENTRIES_FILE_PATH, encoding=sys.getdefaultencoding(),
        errors='replace') as f:
      entries = f.read().splitlines()
    test_sentence = ''.join(self.test_entry.split(utils.SEP))
    self.assertEqual(
        len(entries), len(test_sentence),
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
    os.remove(SOURCE_FILE_PATH)
    os.remove(ENTRIES_FILE_PATH)


if __name__ == '__main__':
  unittest.main()
