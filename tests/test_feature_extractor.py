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

import unittest
import os
import sys
from pathlib import Path

LIB_PATH = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(LIB_PATH))

from budoux import feature_extractor
from budoux import utils

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

SOURCE_FILE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'source_test.txt'))
ENTRIES_FILE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'entries_test.txt'))


class TestFeatureExtractor(unittest.TestCase):

  def setUp(self):
    Path(ENTRIES_FILE_PATH).touch()
    self.test_entry = f'これは{utils.SEP}美しい{utils.SEP}ペンです。'
    with open(SOURCE_FILE_PATH, 'w', encoding=sys.getdefaultencoding()) as f:
      f.write(self.test_entry)

  def test_unicode_block_index(self):

    def check(character, block):
      self.assertEqual(feature_extractor.unicode_block_index(character), block)

    check('a', 1)  # 'a' falls the 1st block 'Basic Latin'.
    check('あ', 108)  # 'あ' falls the 108th block 'Hiragana'.
    check('安', 120)  # '安' falls the 120th block 'Kanji'.

  def test_get_feature(self):
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

    feature = feature_extractor.get_feature('', 'a', 'a', 'a', 'a', 'a', 'a',
                                            'a', 'a')
    self.assertIn('UW1:', feature,
                  'The word feature should be blank for a blank string.')
    self.assertIn(
        'UB1:999', feature,
        'The Unicode block feature should be 999 for a blank string.')

    feature = feature_extractor.get_feature('a', 'a', 'a', '', '', '', 'b', 'b',
                                            'b')
    self.assertNotIn(
        'UW4:', feature,
        'UW features that imply the end of line should not be included.')
    self.assertNotIn(
        'UB4:999', feature,
        'UB features that imply the end of line should not be included.')
    self.assertNotIn(
        'BB3:999999', feature,
        'BB features that imply the end of line should not be included.')

  def test_process(self):
    feature_extractor.process(SOURCE_FILE_PATH, ENTRIES_FILE_PATH)
    with open(
        ENTRIES_FILE_PATH, encoding=sys.getdefaultencoding(),
        errors='replace') as f:
      entries = f.read().splitlines()
    test_sentence = ''.join(self.test_entry.split(utils.SEP))
    self.assertEqual(
        len(entries),
        len(test_sentence) - 2,
        'The first two characters\' ends should not be examined.')

    print(entries)
    labels = [int(entry.split('\t')[0]) for entry in entries]
    self.assertListEqual(
        labels,
        [
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
    self.assertIn('UW3:美', features[1])
    self.assertIn('UW3:し', features[2])
    self.assertIn('UW3:い', features[3])
    self.assertIn('UW3:。', features[-1])

  def tearDown(self):
    os.remove(SOURCE_FILE_PATH)
    os.remove(ENTRIES_FILE_PATH)


if __name__ == '__main__':
  unittest.main()
