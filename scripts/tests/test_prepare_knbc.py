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
"""Tests the prepare KNBC script."""

import os
import sys
import unittest

# module hack
LIB_PATH = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.abspath(LIB_PATH))

from scripts import prepare_knbc  # noqa (module hack)


class TestBreakBeforeSequence(unittest.TestCase):

  def test_standard(self) -> None:
    chunks = ['abcdef', 'ghi']
    result = prepare_knbc.break_before_sequence(chunks, 'de')
    self.assertListEqual(result, ['abc', 'def', 'ghi'])

  def test_sequence_on_top(self) -> None:
    chunks = ['abcdef', 'ghi']
    result = prepare_knbc.break_before_sequence(chunks, 'gh')
    self.assertListEqual(result, ['abcdef', 'ghi'])

  def test_multiple_hit(self) -> None:
    chunks = ['abcabc', 'def']
    result = prepare_knbc.break_before_sequence(chunks, 'bc')
    self.assertListEqual(result, ['a', 'bca', 'bc', 'def'])


class TestKNBCHTMLParser(unittest.TestCase):
  example_html = '''
  <html>
    <body>
      <table>
        <tr><th>HA</th><th>HB</th><th>HC</th><th>HD</th><th>HE</th></tr>
        <tr><td colspan="5" id="bnst-kugiri"><a>文節区切り</a></td></tr>
        <tr><td>abc</td><td></td><td></td><td></td><td></td></tr>
        <tr><td>de</td><td></td><td></td><td></td><td></td></tr>
        <tr><td colspan="5" id="tag-kugiri"><a>タグ区切り</a></td></tr>
        <tr><td>fgh</td><td></td><td></td><td></td><td> </td></tr>
        <tr><td>ijkl</td><td></td><td></td><td></td><td> </td></tr>
        <tr><td colspan="5" id="bnst-kugiri"><a>文節区切り</a></td></tr>
        <tr><td>mn</td><td></td><td></td><td></td><td> </td></tr>
      </table>
    </body>
  </html>
  '''

  def test_parse_phrase(self) -> None:
    parser = prepare_knbc.KNBCHTMLParser('phrase')
    parser.feed(self.example_html)
    self.assertListEqual(parser.chunks, ['abcdefghijkl', 'mn'])

  def test_parse_tag(self) -> None:
    parser = prepare_knbc.KNBCHTMLParser('tag')
    parser.feed(self.example_html)
    self.assertListEqual(parser.chunks, ['abcde', 'fghijkl', 'mn'])

  def test_parse_word(self) -> None:
    parser = prepare_knbc.KNBCHTMLParser('word')
    parser.feed(self.example_html)
    self.assertListEqual(parser.chunks, ['abc', 'de', 'fgh', 'ijkl', 'mn'])
