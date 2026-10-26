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
import tempfile
import unittest

from scripts import prepare_knbc


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


class TestProcessKNBC(unittest.TestCase):
  def test_utf8_html_reading(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      html_dir = os.path.join(temp_dir, 'html')
      os.makedirs(html_dir, exist_ok=True)
      html_path = os.path.join(html_dir, 'sample-morph.html')
      content = '''
      <html>
        <body>
          <table>
            <tr><th>HA</th><th>HB</th><th>HC</th><th>HD</th><th>HE</th></tr>
            <tr><td colspan="5" id="bnst-kugiri"><a>文節区切り</a></td></tr>
            <tr><td>今日</td><td></td><td></td><td></td><td></td></tr>
            <tr><td colspan="5" id="bnst-kugiri"><a>文節区切り</a></td></tr>
            <tr><td>は</td><td></td><td></td><td></td><td></td></tr>
          </table>
        </body>
      </html>
      '''
      with open(html_path, 'w', encoding='utf-8') as f:
        f.write(content)
      outfile = os.path.join(temp_dir, 'output.txt')
      prepare_knbc.process_knbc(temp_dir, outfile)
      with open(outfile, encoding='utf-8') as f:
        lines = f.readlines()
      self.assertEqual(lines, [f'今日{prepare_knbc.utils.SEP}は\n'])
