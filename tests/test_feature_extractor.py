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

  def test_get_feature(self) -> None:
    feature = feature_extractor.get_feature('a', 'b', 'c', 'd', 'e', 'f')
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

            # Bigram of Words (BW)
            'BW1:bc',
            'BW2:cd',
            'BW3:de',

            # Trigram of Words (TW)
            'TW1:abc',
            'TW2:bcd',
            'TW3:cde',
            'TW4:def',
        },
        'Features should be extracted.')

    def find_by_prefix(prefix: str, feature: typing.List[str]) -> bool:
      for item in feature:
        if item.startswith(prefix):
          return True
      return False

    feature = feature_extractor.get_feature('a', 'a', utils.INVALID, 'a', 'a',
                                            'a')
    self.assertFalse(
        find_by_prefix('UW3:', feature),
        'Should omit the Unigram feature when the character is invalid.')
    self.assertFalse(
        find_by_prefix('BW2:', feature),
        'Should omit the Bigram feature that covers an invalid character.')


if __name__ == '__main__':
  unittest.main()
