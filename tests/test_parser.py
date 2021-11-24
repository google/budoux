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

import unittest
import os
import sys
import xml.etree.ElementTree as ET
import html5lib

LIB_PATH = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(LIB_PATH))

from budoux import parser

html_parser = html5lib.HTMLParser()


def compare_html_string(a, b):
  a_normalized = ET.tostring(html_parser.parse(a))
  b_normalized = ET.tostring(html_parser.parse(b))
  return a_normalized == b_normalized


class TestTextContentExtractor(unittest.TestCase):

  def test_output(self):
    input = '<p><a href="#">Hello</a>, <b>World</b></p>'
    expected = 'Hello, World'
    extractor = parser.TextContentExtractor()
    extractor.feed(input)
    self.assertEqual(
        extractor.output, expected,
        'Text content should be extacted from the given HTML string.')


class TestHTMLChunkResolver(unittest.TestCase):

  def test_output(self):
    input = '<p>ab<b>cde</b>f</p>'
    expected = '<p>ab<b>c<wbr>de</b>f</p>'
    resolver = parser.HTMLChunkResolver(['abc', 'def'])
    resolver.feed(input)
    self.assertTrue(
        compare_html_string(resolver.output, expected),
        'WBR tags should be inserted as specified by chunks.')


class TestParser(unittest.TestCase):
  TEST_SENTENCE = 'abcdeabcd'

  def test_parse(self):
    p = parser.Parser({
        'UW4:a': 10000,  # means "should separate right before 'a'".
    })
    chunks = p.parse(TestParser.TEST_SENTENCE)
    self.assertListEqual(chunks, ['abcde', 'abcd'],
                         'should separate but not the first two characters.')

    p = parser.Parser({
        'BP2:UU': 10000,
    })
    chunks = p.parse(TestParser.TEST_SENTENCE)
    self.assertListEqual(
        chunks, ['abc', 'deabcd'],
        'should respect the results feature with a high score.')

    p = parser.Parser({
        'UW4:a': 10,
    })
    chunks = p.parse(TestParser.TEST_SENTENCE, 100)
    self.assertListEqual(
        chunks, [TestParser.TEST_SENTENCE],
        'should ignore features with scores lower than the threshold.')

    p = parser.Parser({})
    chunks = p.parse('')
    self.assertListEqual(chunks, [],
                         'should return a blank list when the input is blank.')

  def test_translate_html_string(self):
    p = parser.Parser({
        'UW4:a': 10000,  # means "should separate right before 'a'".
    })

    input_html = 'xyzabcd'
    expected_html = (
        '<span style="word-break: keep-all; overflow-wrap: break-word;">'
        'xyz<wbr>abcd</span>')
    output_html = p.translate_html_string(input_html)
    self.assertTrue(
        compare_html_string(output_html, expected_html),
        'should output a html string with a SPAN parent with proper style attributes.'
    )

    input_html = 'xyz<script>alert(1);</script>xyzabc'
    expected_html = (
        '<span style="word-break: keep-all; overflow-wrap: break-word;">'
        'xyz<script>alert(1);</script>xyz<wbr>abc</span>')
    output_html = p.translate_html_string(input_html)
    self.assertTrue(
        compare_html_string(output_html, expected_html),
        'should pass script tags as is.')

    input_html = 'xyz<code>abc</code>abc'
    expected_html = (
        '<span style="word-break: keep-all; overflow-wrap: break-word;">'
        'xyz<code>abc</code><wbr>abc</span>')
    output_html = p.translate_html_string(input_html)
    self.assertTrue(
        compare_html_string(output_html, expected_html),
        'should skip some specific tags.')

    input_html = 'xyza<a href="#" hidden>bc</a>abc'
    expected_html = (
        '<span style="word-break: keep-all; overflow-wrap: break-word;">'
        'xyz<wbr>a<a href="#" hidden>bc</a><wbr>abc</span>')
    output_html = p.translate_html_string(input_html)
    self.assertTrue(
        compare_html_string(output_html, expected_html),
        'should not ruin attributes of child elements.')

    input_html = 'xyza🇯🇵🇵🇹abc'
    expected_html = (
        '<span style="word-break: keep-all; overflow-wrap: break-word;">'
        'xyz<wbr>a🇯🇵🇵🇹<wbr>abc</span>')
    output_html = p.translate_html_string(input_html)
    self.assertTrue(
        compare_html_string(output_html, expected_html),
        'should work with emojis.')


if __name__ == '__main__':
  unittest.main()