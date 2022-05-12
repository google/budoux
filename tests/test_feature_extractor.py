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

import typing
import unittest

from budoux import feature_extractor, utils


class TestFeatureExtractor(unittest.TestCase):

  def test_unicode_block_index(self) -> None:

    def check(character: str, block: str, msg: str) -> None:
      self.assertEqual(
          feature_extractor.unicode_block_index(character), block, msg)

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


if __name__ == '__main__':
  unittest.main()
