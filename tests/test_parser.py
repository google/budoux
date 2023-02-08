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

import os
import sys
import unittest
import xml.etree.ElementTree as ET

import html5lib

# module hack
LIB_PATH = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(LIB_PATH))

from budoux import parser  # noqa (module hack)

html_parser = html5lib.HTMLParser()


def compare_html_string(a: str, b: str) -> bool:
  a_normalized = ET.tostring(html_parser.parse(a))
  b_normalized = ET.tostring(html_parser.parse(b))
  return a_normalized == b_normalized


class TestTextContentExtractor(unittest.TestCase):

  def test_output(self) -> None:
    input = '<p><a href="#">Hello</a>, <b>World</b></p>'
    expected = 'Hello, World'
    extractor = parser.TextContentExtractor()
    extractor.feed(input)
    self.assertEqual(
        extractor.output, expected,
        'Text content should be extacted from the given HTML string.')


class TestHTMLChunkResolver(unittest.TestCase):

  def test_output(self) -> None:
    input = '<p>ab<b>cde</b>f</p>'
    expected = '<p>ab<b>c<wbr>de</b>f</p>'
    resolver = parser.HTMLChunkResolver(['abc', 'def'])
    resolver.feed(input)
    self.assertTrue(
        compare_html_string(resolver.output, expected),
        'WBR tags should be inserted as specified by chunks.')


class TestParser(unittest.TestCase):
  TEST_SENTENCE = 'abcdeabcd'

  def test_parse(self) -> None:
    p = parser.Parser({
        'UW4': {
            'a': 10000
        },  # means "should separate right before 'a'".
    })
    chunks = p.parse(TestParser.TEST_SENTENCE)
    self.assertListEqual(chunks, ['abcde', 'abcd'],
                         'Should separate if a strong feature item supports.')

    p = parser.Parser({
        'UW4': {
            'b': 10000
        },  # means "should separate right before 'b'".
    })
    chunks = p.parse(TestParser.TEST_SENTENCE)
    self.assertListEqual(
        chunks, ['a', 'bcdea', 'bcd'],
        'Should separate even if it makes the first character a sole phrase.')

    p = parser.Parser({})
    chunks = p.parse('')
    self.assertListEqual(chunks, [],
                         'Should return a blank list when the input is blank.')

  def test_translate_html_string(self) -> None:
    p = parser.Parser({
        'UW4': {
            'a': 10000
        },  # means "should separate right before 'a'".
    })

    input_html = 'xyzabcd'
    expected_html = (
        '<span style="word-break: keep-all; overflow-wrap: break-word;">'
        'xyz<wbr>abcd</span>')
    output_html = p.translate_html_string(input_html)
    self.assertTrue(
        compare_html_string(output_html, expected_html),
        'Should output a html string with a SPAN parent with proper style attributes.'
    )

    input_html = 'xyz<script>alert(1);</script>xyzabc'
    expected_html = (
        '<span style="word-break: keep-all; overflow-wrap: break-word;">'
        'xyz<script>alert(1);</script>xyz<wbr>abc</span>')
    output_html = p.translate_html_string(input_html)
    self.assertTrue(
        compare_html_string(output_html, expected_html),
        'Should pass script tags as is.')

    input_html = 'xyz<code>abc</code>abc'
    expected_html = (
        '<span style="word-break: keep-all; overflow-wrap: break-word;">'
        'xyz<code>abc</code><wbr>abc</span>')
    output_html = p.translate_html_string(input_html)
    self.assertTrue(
        compare_html_string(output_html, expected_html),
        'Should skip some specific tags.')

    input_html = 'xyza<a href="#" hidden>bc</a>abc'
    expected_html = (
        '<span style="word-break: keep-all; overflow-wrap: break-word;">'
        'xyz<wbr>a<a href="#" hidden>bc</a><wbr>abc</span>')
    output_html = p.translate_html_string(input_html)
    self.assertTrue(
        compare_html_string(output_html, expected_html),
        'Should not ruin attributes of child elements.')

    input_html = 'xyza🇯🇵🇵🇹abc'
    expected_html = (
        '<span style="word-break: keep-all; overflow-wrap: break-word;">'
        'xyz<wbr>a🇯🇵🇵🇹<wbr>abc</span>')
    output_html = p.translate_html_string(input_html)
    self.assertTrue(
        compare_html_string(output_html, expected_html),
        'Should work with emojis.')


class TestDefaultParser(unittest.TestCase):

  def test_load_default_japanese_parser(self) -> None:
    p_ja = parser.load_default_japanese_parser()
    phrases = p_ja.parse('Google の使命は、世界中の情報を整理し、世界中の人がアクセスできて使えるようにすることです。')
    self.assertListEqual(phrases, [
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
    ])

  def test_load_default_simplified_chinese_parser(self) -> None:
    p_hans = parser.load_default_simplified_chinese_parser()
    phrases = p_hans.parse('我们的使命是整合全球信息，供大众使用，让人人受益。')
    self.assertListEqual(phrases, [
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
        '人',
        '人',
        '受益。',
    ])

  def test_load_default_traditional_chinese_parser(self) -> None:
    p_hant = parser.load_default_traditional_chinese_parser()
    phrases = p_hant.parse('我們的使命是匯整全球資訊，供大眾使用，使人人受惠。')
    self.assertListEqual(phrases, [
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
        '人',
        '人',
        '受惠。',
    ])


if __name__ == '__main__':
  unittest.main()
