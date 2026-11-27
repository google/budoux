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
"""Tests the BudouX parser."""

import unicodedata
import unittest

from budoux import parser


class TestParser(unittest.TestCase):
  TEST_SENTENCE = 'abcdeabcd'

  def test_parse(self) -> None:
    p = parser.Parser(
      {
        'UW4': {'a': 10000}  # means "should separate right before 'a'".
      }
    )
    chunks = p.parse(TestParser.TEST_SENTENCE)
    self.assertListEqual(
      chunks, ['abcde', 'abcd'], 'Should separate if a strong feature item supports.'
    )

    p = parser.Parser(
      {
        'UW4': {'b': 10000}  # means "should separate right before 'b'".
      }
    )
    chunks = p.parse(TestParser.TEST_SENTENCE)
    self.assertListEqual(
      chunks,
      ['a', 'bcdea', 'bcd'],
      'Should separate even if it makes the first character a sole phrase.',
    )

    p = parser.Parser({})
    chunks = p.parse('')
    self.assertListEqual(
      chunks, [], 'Should return a blank list when the input is blank.'
    )

  def test_parse_boundaries_and_defensive_copy(self) -> None:
    raw_model = {'UW4': {'a': 10000}}
    p = parser.Parser(raw_model)
    raw_model['UW4']['a'] = -10000
    self.assertListEqual(p.parse_boundaries('abcdeabcd'), [5])

  def test_invisible_format_chars_idempotency(self) -> None:
    p = parser.Parser({'UW4': {'d': 1000, 'e': 1000}, 'UW2': {'c': -2000}})
    once = p.translate_html_string('abcdef', wrap=False)
    self.assertEqual(once, 'abc\u200bdef')
    twice = p.translate_html_string(once, wrap=False)
    self.assertEqual(twice, once)
    self.assertListEqual(p.parse_boundaries('abc\u200bdef'), [4])
    self.assertListEqual(p.parse('\u200b'), ['\u200b'])

  def test_no_boundary_inside_grapheme_cluster(self) -> None:
    # A negative total score places a boundary before every character.
    p = parser.Parser({'UW1': {'x': -1}})
    self.assertListEqual(p.parse('abc'), ['a', 'b', 'c'])
    cases = [
      # か + U+3099 COMBINING KATAKANA-HIRAGANA VOICED SOUND MARK
      ('か\u3099ら', ['か\u3099', 'ら'], 'combining dakuten'),
      # ก + U+0E35 THAI CHARACTER SARA II
      ('ก\u0e35a', ['ก\u0e35', 'a'], 'Thai vowel sign'),
      ('a❤\ufe0fb', ['a', '❤\ufe0f', 'b'], 'variation selector'),
      ('葛\U000e0100飾', ['葛\U000e0100', '飾'], 'IVS'),
      ('1\ufe0f\u20e3a', ['1\ufe0f\u20e3', 'a'], 'keycap'),
      ('\U0001f44d\U0001f3fda', ['\U0001f44d\U0001f3fd', 'a'], 'emoji modifier'),
      (
        '\U0001f468\u200d\U0001f469\u200d\U0001f467a',
        ['\U0001f468\u200d\U0001f469\u200d\U0001f467', 'a'],
        'ZWJ sequence',
      ),
      (
        '\U0001f1ef\U0001f1f5\U0001f1f9\U0001f1fc\U0001f1ef',
        ['\U0001f1ef\U0001f1f5', '\U0001f1f9\U0001f1fc', '\U0001f1ef'],
        'regional indicator pairs',
      ),
      # Not General_Category M, but UAX #29 Extend or SpacingMark.
      (
        '\uff76\uff9e\uff8a\uff9f',
        ['\uff76\uff9e', '\uff8a\uff9f'],
        'halfwidth (semi-)voiced marks',
      ),
      ('\u0e01\u0e33a', ['\u0e01\u0e33', 'a'], 'Thai SARA AM'),
      ('\u0e81\u0eb3a', ['\u0e81\u0eb3', 'a'], 'Lao SARA AM'),
      ('a\u200cb', ['a\u200c', 'b'], 'ZWNJ'),
      (
        '\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007fa',
        ['\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f', 'a'],
        'emoji tag sequence',
      ),
      ('a\r\nb', ['a', '\r\n', 'b'], 'CR LF'),
      ('a\nb', ['a', '\n', 'b'], 'LF without CR'),
      # An invisible format character in between ends the cluster.
      ('\u304b\u200b\u3099', ['\u304b\u200b', '\u3099'], 'mark after ZWSP'),
      (
        '\U0001f1ef\u200b\U0001f1f5\U0001f1f7',
        ['\U0001f1ef\u200b', '\U0001f1f5\U0001f1f7'],
        'regional indicators across ZWSP',
      ),
    ]
    for sentence, expected, name in cases:
      with self.subTest(name):
        self.assertListEqual(p.parse(sentence), expected)

  def test_translate_html_string(self) -> None:
    p = parser.Parser(
      {
        'UW4': {'a': 10000}  # means "should separate right before 'a'".
      }
    )

    input_html = 'xyzabcd'
    expected_html = (
      '<span style="word-break: keep-all; overflow-wrap: anywhere;">'
      'xyz\u200babcd</span>'
    )
    output_html = p.translate_html_string(input_html)
    self.assertEqual(
      output_html,
      expected_html,
      'Should output a html string with a SPAN parent with proper style attributes.',
    )

    input_html = 'xyz<script>alert(1);</script>xyzabc'
    expected_html = (
      '<span style="word-break: keep-all; overflow-wrap: anywhere;">'
      'xyz<script>alert(1);</script>xyz\u200babc</span>'
    )
    output_html = p.translate_html_string(input_html)
    self.assertEqual(output_html, expected_html, 'Should pass script tags as is.')

    input_html = 'xyz<code>abc</code>abc'
    expected_html = (
      '<span style="word-break: keep-all; overflow-wrap: anywhere;">'
      'xyz\u200b<code>abc</code>\u200babc</span>'
    )
    output_html = p.translate_html_string(input_html)
    self.assertEqual(output_html, expected_html, 'Should skip some specific tags.')

    input_html = 'xyza<a href="#" hidden>bc</a>abc'
    expected_html = (
      '<span style="word-break: keep-all; overflow-wrap: anywhere;">'
      'xyz\u200ba<a href="#" hidden>bc</a>\u200babc</span>'
    )
    output_html = p.translate_html_string(input_html)
    self.assertEqual(
      output_html, expected_html, 'Should not ruin attributes of child elements.'
    )

    input_html = 'xyza🇯🇵🇵🇹abc'
    expected_html = (
      '<span style="word-break: keep-all; overflow-wrap: anywhere;">'
      'xyz\u200ba🇯🇵🇵🇹\u200babc</span>'
    )
    output_html = p.translate_html_string(input_html)
    self.assertEqual(output_html, expected_html, 'Should work with emojis.')

    input_html = 'xyzab&c'
    expected_html = (
      '<span style="word-break: keep-all; overflow-wrap: anywhere;">'
      'xyz\u200bab&amp;c</span>'
    )
    output_html = p.translate_html_string(input_html)
    self.assertEqual(
      output_html, expected_html, 'Should keep trailing text with an ampersand.'
    )

    input_html = 'xyza<a href="#" hidden>bc</a>abc'
    expected_html = 'xyz\u200ba<a href="#" hidden>bc</a>\u200babc'
    output_html = p.translate_html_string(input_html, wrap=False)
    self.assertEqual(
      output_html, expected_html, 'Should omit the wrapping SPAN when wrap=False.'
    )


class TestDefaultParser(unittest.TestCase):
  def test_load_default_japanese_parser(self) -> None:
    p_ja = parser.load_default_japanese_parser()
    phrases = p_ja.parse(
      'Google の使命は、世界中の情報を整理し、世界中の人がアクセスできて使えるようにすることです。'
    )
    self.assertListEqual(
      phrases,
      [
        'Google の',
        '使命は、',
        '世界中の',
        '情報を',
        '整理し、',
        '世界中の',
        '人が',
        'アクセスできて',
        '使えるように',
        'する',
        'ことです。',
      ],
    )

  def test_japanese_parser_decomposed_text(self) -> None:
    p_ja = parser.load_default_japanese_parser()
    sentence = unicodedata.normalize('NFD', '今日はとても良い天気です。')
    phrases = p_ja.parse(sentence)
    self.assertEqual(''.join(phrases), sentence)
    for phrase in phrases:
      self.assertFalse(
        unicodedata.category(phrase[0]).startswith('M'),
        f'A phrase starts with a combining mark: {phrases!r}',
      )

  def test_load_default_simplified_chinese_parser(self) -> None:
    p_hans = parser.load_default_simplified_chinese_parser()
    phrases = p_hans.parse('我们的使命是整合全球信息，供大众使用，让人人受益。')
    self.assertListEqual(
      phrases,
      [
        '我们',
        '的',
        '使命',
        '是',
        '整合',
        '全球',
        '信息，',
        '供',
        '大众',
        '使用，',
        '让',
        '人人',
        '受益。',
      ],
    )

  def test_load_default_traditional_chinese_parser(self) -> None:
    p_hant = parser.load_default_traditional_chinese_parser()
    phrases = p_hant.parse('我們的使命是匯整全球資訊，供大眾使用，使人人受惠。')
    self.assertListEqual(
      phrases,
      [
        '我們',
        '的',
        '使命',
        '是',
        '匯整',
        '全球',
        '資訊，',
        '供',
        '大眾',
        '使用，',
        '使',
        '人人',
        '受惠。',
      ],
    )

  def test_load_default_thai_parser(self) -> None:
    p_th = parser.load_default_thai_parser()
    phrases = p_th.parse('วันนี้อากาศดี')
    self.assertListEqual(phrases, ['วัน', 'นี้', 'อากาศ', 'ดี'])
