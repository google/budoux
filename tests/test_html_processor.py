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
"Tests the HTML Processor."

import unittest

from budoux import html_processor
from budoux.parser import Parser


def _break_before(*targets: str) -> Parser:
  """Creates a deterministic parser that breaks right before each target char."""
  model: dict[str, dict[str, int]] = {'UW4': dict.fromkeys(targets, 1000)}
  if len(targets) > 1:
    model['UW1'] = {'\u0000': -(len(targets) - 1) * 1000}
  return Parser(model)


def _break_after(*targets: str) -> Parser:
  """Creates a deterministic parser that breaks right after each target char."""
  model: dict[str, dict[str, int]] = {'UW3': dict.fromkeys(targets, 1000)}
  if len(targets) > 1:
    model['UW1'] = {'\u0000': -(len(targets) - 1) * 1000}
  return Parser(model)


def _break_between_pairs(*pairs: str) -> Parser:
  """Creates a deterministic parser that breaks between adjacent 2-char pairs."""
  model: dict[str, dict[str, int]] = {'BW2': dict.fromkeys(pairs, 1000)}
  if len(pairs) > 1:
    model['UW1'] = {'\u0000': -(len(pairs) - 1) * 1000}
  return Parser(model)


def _break_everywhere() -> Parser:
  """Creates a deterministic parser that breaks at every character boundary."""
  return Parser({'UW4': {'\u0000': -10000}})


class TestHTMLChunkResolver(unittest.TestCase):
  def test_output(self) -> None:
    input_html = '<p>ab<b>cde</b>f</p>'
    expected = '<p>ab<b>c<wbr>de</b>f</p>'
    result = html_processor.translate(
      _break_before('d'), input_html, '<wbr>', wrap=False
    )
    self.assertEqual(
      result, expected, 'WBR tags should be inserted as specified by chunks.'
    )

  def test_unpaired(self) -> None:
    input_html = '<p>abcdef</p></p>'
    expected = '<p>abc<wbr>def</p></p>'
    result = html_processor.translate(
      _break_before('d'), input_html, '<wbr>', wrap=False
    )
    self.assertEqual(result, expected, 'Unpaired close tag should not cause errors.')

  def test_nobr(self) -> None:
    input_html = '<p>ab<nobr>cde</nobr>f</p>'
    expected = '<p>ab<nobr>cde</nobr>f</p>'
    result = html_processor.translate(
      _break_before('d'), input_html, '<wbr>', wrap=False
    )
    self.assertEqual(result, expected, 'WBR tags should not be inserted if in NOBR.')

  def test_unmatched_end_tag_preserves_skip_state(self) -> None:
    result = html_processor.translate(
      _break_before('d', 'g'), '<nobr>abc</span>def</nobr>ghi', '<wbr>', wrap=False
    )
    self.assertEqual(result, '<nobr>abc</span>def</nobr><wbr>ghi')

  def test_unmatched_end_tag_preserves_parent_stack(self) -> None:
    parser = _break_before('d', 'g', 'j')
    self.assertEqual(
      html_processor.translate(
        parser, '<b><nobr>abc</span>def</b>ghi', '<wbr>', wrap=False
      ),
      '<b><nobr>abc</span>def</b><wbr>ghi',
    )
    self.assertEqual(
      html_processor.translate(
        parser, '<div><nobr>abc</span>def</div>ghijkl', '<wbr>', wrap=False
      ),
      '<div><nobr>abc</span>def</div>ghi<wbr>jkl',
    )

  def test_after_nobr(self) -> None:
    input_html = '<p>ab<nobr>xy</nobr>abcdef</p>'
    expected = '<p>ab<nobr>xy</nobr>abc<wbr>def</p>'
    result = html_processor.translate(
      _break_before('d'), input_html, '<wbr>', wrap=False
    )
    self.assertEqual(result, expected, 'WBR tags should be inserted if after NOBR.')

  def test_img_in_nobr(self) -> None:
    input_html = '<p>ab<nobr>x<img>y</nobr>abcdef</p>'
    expected = '<p>ab<nobr>x<img>y</nobr>abc<wbr>def</p>'
    result = html_processor.translate(
      _break_before('d'), input_html, '<wbr>', wrap=False
    )
    self.assertEqual(result, expected, 'IMG should not affect surrounding NOBR.')


class TestResolve(unittest.TestCase):
  def test_with_simple_text_input(self) -> None:
    parser = _break_before('d')
    html = 'abcdef'
    result = html_processor.translate(parser, html)
    expected = (
      '<span style="word-break: keep-all; overflow-wrap: anywhere;">abc\u200bdef</span>'
    )
    self.assertEqual(result, expected)

  def test_with_standard_html_input(self) -> None:
    parser = _break_before('d')
    html = 'ab<a href="http://example.com">cd</a>ef'
    result = html_processor.translate(parser, html)
    expected = '<span style="word-break: keep-all; overflow-wrap: anywhere;">ab<a href="http://example.com">c\u200bd</a>ef</span>'
    self.assertEqual(result, expected)

  def test_with_escaped_characters(self) -> None:
    parser = _break_before('c')
    html = '<a title="&quot;x&quot;">a&lt;b&gt;c&amp;d</a>'
    result = html_processor.translate(parser, html)
    expected = '<span style="word-break: keep-all; overflow-wrap: anywhere;"><a title="&quot;x&quot;">a&lt;b&gt;\u200bc&amp;d</a></span>'
    self.assertEqual(result, expected)

  def test_with_escaped_characters_in_rcdata_element(self) -> None:
    parser = _break_before('c')
    html = '<textarea>a&lt;"bc&amp;d</textarea>'
    result = html_processor.translate(parser, html)
    expected = '<span style="word-break: keep-all; overflow-wrap: anywhere;"><textarea>a&lt;"bc&amp;d</textarea></span>'
    self.assertEqual(result, expected)

  def test_with_nodes_to_skip(self) -> None:
    parser = _break_before('g')
    html = 'a<button>bcde</button>fghi'
    result = html_processor.translate(parser, html)
    expected = '<span style="word-break: keep-all; overflow-wrap: anywhere;">a<button>bcde</button>f\u200bghi</span>'
    self.assertEqual(result, expected)

  def test_with_select_skip_node(self) -> None:
    parser = _break_before('g')
    html = '<select><option>a</option><option>bcde</option></select>fghi'
    result = html_processor.translate(parser, html)
    expected = (
      '<span style="word-break: keep-all; overflow-wrap: anywhere;">'
      '<select><option>a</option><option>bcde</option></select>f\u200bghi</span>'
    )
    self.assertEqual(result, expected)

  def test_with_pre_skip_node(self) -> None:
    parser = _break_before('g')
    html = '<pre>abcde</pre>fghi'
    result = html_processor.translate(parser, html)
    expected = (
      '<span style="word-break: keep-all; overflow-wrap: anywhere;">'
      '<pre>abcde</pre>f\u200bghi</span>'
    )
    self.assertEqual(result, expected)

  def test_with_break_before_skip(self) -> None:
    parser = _break_before('d', 'g', 'j')
    self.assertEqual(
      html_processor.translate(parser, 'abc<nobr>defghi</nobr>jkl'),
      '<span style="word-break: keep-all; overflow-wrap: anywhere;">'
      'abc\u200b<nobr>defghi</nobr>\u200bjkl</span>',
    )
    self.assertEqual(
      html_processor.translate(parser, 'abc<button>defghi</button>jkl'),
      '<span style="word-break: keep-all; overflow-wrap: anywhere;">'
      'abc<button>defghi</button>\u200bjkl</span>',
    )

  def test_with_break_before_skip_preceded_by_whitespace(self) -> None:
    parser = _break_before('d', 'g')
    self.assertEqual(
      html_processor.translate(parser, 'abc <button>def</button>ghi'),
      '<span style="word-break: keep-all; overflow-wrap: anywhere;">'
      'abc <button>def</button>ghi</span>',
    )
    self.assertEqual(
      html_processor.translate(parser, 'abc <nobr>def</nobr>ghi'),
      '<span style="word-break: keep-all; overflow-wrap: anywhere;">'
      'abc <nobr>def</nobr>\u200bghi</span>',
    )

  def test_with_break_before_skip_starting_with_whitespace(self) -> None:
    parser = _break_before(' ', 'd', 'g')
    self.assertEqual(
      html_processor.translate(parser, 'abc<button> def</button>ghi'),
      '<span style="word-break: keep-all; overflow-wrap: anywhere;">'
      'abc<button> def</button>\u200bghi</span>',
    )
    self.assertEqual(
      html_processor.translate(parser, 'abc<nobr> def</nobr>ghi'),
      '<span style="word-break: keep-all; overflow-wrap: anywhere;">'
      'abc<nobr> def</nobr>\u200bghi</span>',
    )

  def test_with_skip_node_at_end(self) -> None:
    parser = _break_before('d')
    html = 'abcdef<button></button>'
    result = html_processor.translate(parser, html)
    expected = '<span style="word-break: keep-all; overflow-wrap: anywhere;">abc\u200bdef<button></button></span>'
    self.assertEqual(result, expected)

  def test_with_nothing_to_split(self) -> None:
    parser = Parser({})
    html = 'abcdef'
    result = html_processor.translate(parser, html)
    expected = (
      '<span style="word-break: keep-all; overflow-wrap: anywhere;">abcdef</span>'
    )
    self.assertEqual(result, expected)

  def test_with_list_items_and_whitespace(self) -> None:
    parser = _break_before('d')
    html = '<ul><li>abc</li>\n<li>def</li></ul>'
    result = html_processor.translate(parser, html, '<wbr>')
    expected = '<span style="word-break: keep-all; overflow-wrap: anywhere;"><ul><li>abc</li>\n<li>def</li></ul></span>'
    self.assertEqual(result, expected)

  def test_with_break_after_whitespace_in_list(self) -> None:
    parser = _break_after('c')
    html = '<ul><li>abc</li>\n<li>def</li></ul>'
    result = html_processor.translate(parser, html, '<wbr>')
    expected = '<span style="word-break: keep-all; overflow-wrap: anywhere;"><ul><li>abc</li>\n<li>def</li></ul></span>'
    self.assertEqual(result, expected)

  def test_with_self_closing_tag(self) -> None:
    parser = _break_before('d', 'e')
    html = 'abc<br/>def'
    result = html_processor.translate(parser, html)
    expected = '<span style="word-break: keep-all; overflow-wrap: anywhere;">abc<br/>d\u200bef</span>'
    self.assertEqual(result, expected)

  def test_with_self_closing_skip_node(self) -> None:
    parser = _break_before('d', 'g')
    html = 'abc<input type="text" />defghi'
    result = html_processor.translate(parser, html)
    expected = '<span style="word-break: keep-all; overflow-wrap: anywhere;">abc<input type="text" />\u200bdef\u200bghi</span>'
    self.assertEqual(result, expected)

  def test_with_void_skip_node(self) -> None:
    parser = _break_before('d', 'g')
    html = 'abc<input type="text">defghi'
    result = html_processor.translate(parser, html)
    expected = '<span style="word-break: keep-all; overflow-wrap: anywhere;">abc<input type="text">\u200bdef\u200bghi</span>'
    self.assertEqual(result, expected)

  def test_with_stray_void_end_tag_in_skip_node(self) -> None:
    parser = _break_before('d', 'g', 'j')
    html = '<code>abc<br></br>def</code>ghijkl'
    result = html_processor.translate(parser, html)
    expected = '<span style="word-break: keep-all; overflow-wrap: anywhere;"><code>abc<br></br>def</code>\u200bghi\u200bjkl</span>'
    self.assertEqual(result, expected)

  def test_with_sep_char_in_skip_node(self) -> None:
    parser = Parser({})
    html = '<p>abc<code>▁</code>def</p>'
    result = html_processor.translate(parser, html)
    expected = '<span style="word-break: keep-all; overflow-wrap: anywhere;"><p>abc<code>▁</code>def</p></span>'
    self.assertEqual(result, expected)
    self.assertEqual(
      html_processor.translate(_break_before('e'), html, wrap=False),
      '<p>abc<code>▁</code>d\u200bef</p>',
    )

  def test_with_sep_char_at_chunk_boundary(self) -> None:
    parser = _break_before('▁')
    html = '<p>今日は▁天気です。</p>'
    result = html_processor.translate(parser, html)
    expected = '<span style="word-break: keep-all; overflow-wrap: anywhere;"><p>今日は\u200b▁天気です。</p></span>'
    self.assertEqual(result, expected)

  def test_with_existing_separators(self) -> None:
    parser = _break_before('d', 'g')
    html = 'abc<wbr>def\u200bghi'
    result = html_processor.translate(parser, html)
    expected = '<span style="word-break: keep-all; overflow-wrap: anywhere;">abc<wbr>def\u200bghi</span>'
    self.assertEqual(result, expected)

  def test_with_existing_separators_at_start_of_chunk(self) -> None:
    parser = _break_before('d')
    html = 'abc\u200bdef'
    result = html_processor.translate(parser, html)
    expected = (
      '<span style="word-break: keep-all; overflow-wrap: anywhere;">abc\u200bdef</span>'
    )
    self.assertEqual(result, expected)

  def test_with_standalone_existing_separator(self) -> None:
    parser = _break_between_pairs('cd')
    html = 'abc\u200bdef'
    result = html_processor.translate(parser, html)
    expected = (
      '<span style="word-break: keep-all; overflow-wrap: anywhere;">abc\u200bdef</span>'
    )
    self.assertEqual(result, expected)

  def test_with_comments(self) -> None:
    parser = _break_before('d')
    html = '<p>abc<!-- note -->def</p>'
    result = html_processor.translate(parser, html, wrap=False)
    expected = '<p>abc<!-- note -->\u200bdef</p>'
    self.assertEqual(result, expected)

  def test_with_comments_before_text_and_in_skip_node(self) -> None:
    parser = _break_before('d')
    html = '<!-- head --><nobr>abc<!-- x -->def</nobr>'
    result = html_processor.translate(parser, html, wrap=False)
    expected = '<!-- head --><nobr>abc<!-- x -->def</nobr>'
    self.assertEqual(result, expected)

  def test_without_wrap(self) -> None:
    parser = _break_before('d')
    html = 'ab<a href="http://example.com">cd</a>ef'
    result = html_processor.translate(parser, html, wrap=False)
    expected = 'ab<a href="http://example.com">c\u200bd</a>ef'
    self.assertEqual(result, expected)

  def test_block_boundaries_isolate_paragraphs(self) -> None:
    break_before_a = _break_before('a')
    self.assertEqual(
      break_before_a.translate_html_string('<p>xyz</p><p>abc</p>', wrap=False),
      '<p>xyz</p><p>abc</p>',
    )

    lookahead_parser = Parser({'TW4': {'bcd': 1000}})
    self.assertEqual(
      lookahead_parser.translate_html_string('<p>ab</p><p>cd</p>', wrap=False),
      '<p>ab</p><p>cd</p>',
    )

    pair_parser = _break_between_pairs('bc')
    self.assertEqual(
      pair_parser.translate_html_string('<div>ab<div>xy</div>cd</div>', wrap=False),
      '<div>ab<div>xy</div>cd</div>',
    )

  def test_ruby_with_explicit_and_omitted_closing_tags(self) -> None:
    parser = _break_between_pairs('bc', 'xy', '漢字')
    self.assertEqual(
      parser.translate_html_string(
        '<ruby>ab<rp>(</rp><rt>xyz</rt><rp>)</rp></ruby>cd', wrap=False
      ),
      '<ruby>ab<rp>(</rp><rt>xyz</rt><rp>)</rp></ruby>\u200bcd',
    )
    # Omitted </rt> and </rp> tags per HTML5 optional end-tag rules
    self.assertEqual(
      parser.translate_html_string(
        '<ruby>漢<rt>かん<rp>(</rp>字<rt>じ</ruby>', wrap=False
      ),
      '<ruby>漢<rt>かん<rp>(</rp>\u200b字<rt>じ</ruby>',
    )
    self.assertEqual(
      parser.translate_html_string('<ruby>漢字<rt>かんじ</ruby>', wrap=False),
      '<ruby>漢\u200b字<rt>かんじ</ruby>',
    )

  def test_skip_vs_no_break_semantics(self) -> None:
    skip_parser = _break_between_pairs('af')
    self.assertEqual(
      skip_parser.translate_html_string('a<button>bcde</button>fghi', wrap=False),
      'a<button>bcde</button>\u200bfghi',
    )
    self.assertEqual(
      skip_parser.translate_html_string(
        'a<svg><text>bcde</text></svg>fghi', wrap=False
      ),
      'a<svg><text>bcde</text></svg>\u200bfghi',
    )
    self.assertEqual(
      skip_parser.translate_html_string(
        'a<script>if (x < y && z > w) {}</script>fghi', wrap=False
      ),
      'a<script>if (x < y && z > w) {}</script>\u200bfghi',
    )
    # Block-like skip nodes (<pre>, <xmp>, <listing>) also flush the paragraph
    self.assertEqual(
      skip_parser.translate_html_string('a<pre>bcde</pre>fghi', wrap=False),
      'a<pre>bcde</pre>fghi',
    )
    self.assertEqual(
      _break_between_pairs('ef').translate_html_string(
        'ae<pre>bcde</pre>fghi', wrap=False
      ),
      'ae<pre>bcde</pre>fghi',
    )

    no_break_parser = _break_between_pairs('cd', 'de', 'fg')
    for tag in ('code', 'time', 'abbr', 'var', 'kbd', 'samp', 'nobr'):
      self.assertEqual(
        no_break_parser.translate_html_string(f'abc<{tag}>def</{tag}>ghi', wrap=False),
        f'abc\u200b<{tag}>def</{tag}>\u200bghi',
      )
    self.assertEqual(
      _break_everywhere().translate_html_string(
        'ab<nobr>cd\u200b</nobr>ef', wrap=False
      ),
      'a\u200bb\u200b<nobr>cd\u200b</nobr>e\u200bf',
    )

  def test_br_and_hr_break_paragraphs(self) -> None:
    every = _break_everywhere()
    self.assertEqual(
      every.translate_html_string('ab<br>𠮷野家', wrap=False),
      'a\u200bb<br>𠮷\u200b野\u200b家',
    )
    self.assertEqual(
      every.translate_html_string('<div>1<span>23<br>45</span>6</div>', wrap=False),
      '<div>1<span>\u200b2\u200b3<br>4\u200b5</span>\u200b6</div>',
    )
    self.assertEqual(
      every.translate_html_string('ab<hr>cd', wrap=False), 'a\u200bb<hr>c\u200bd'
    )
    self.assertEqual(
      _break_before('天').translate_html_string('今日は<br>天気です。', wrap=False),
      '今日は<br>天気です。',
    )
    self.assertEqual(
      _break_after('c').translate_html_string('abc<br>', wrap=False), 'abc<br>'
    )

  def test_nbsp_and_unicode_space_entity_preservation(self) -> None:
    parser = _break_before('d', 'g')
    # Named &nbsp; and numeric &#160; suppress adjacent breaks while staying verbatim
    self.assertEqual(
      parser.translate_html_string('abc&nbsp;def', wrap=False), 'abc&nbsp;def'
    )
    self.assertEqual(
      parser.translate_html_string('abc&#160;def', wrap=False), 'abc&#160;def'
    )
    self.assertEqual(
      parser.translate_html_string('abc\u00a0def', wrap=False), 'abc\u00a0def'
    )
    self.assertEqual(
      _break_after('\u202f').translate_html_string('abc\u202fdef', wrap=False),
      'abc\u202fdef',
    )
    self.assertEqual(
      parser.translate_html_string('abc&nbsp;<button>def</button>ghi', wrap=False),
      'abc&nbsp;<button>def</button>ghi',
    )
    self.assertEqual(
      parser.translate_html_string('abc&nbsp;<nobr>def</nobr>ghi', wrap=False),
      'abc&nbsp;<nobr>def</nobr>\u200bghi',
    )

  def test_doctype_pi_cdata_and_rawtext_lossless(self) -> None:
    parser = _break_before('d')
    self.assertEqual(
      parser.translate_html_string(
        '<!DOCTYPE html><?xml version="1.0"?><p>abcdef</p>', wrap=False
      ),
      '<!DOCTYPE html><?xml version="1.0"?><p>abc\u200bdef</p>',
    )
    self.assertEqual(
      parser.translate_html_string('<svg><![CDATA[abcdef]]></svg>abcdef', wrap=False),
      '<svg><![CDATA[abcdef]]></svg>abc\u200bdef',
    )
    # &#xffff; decodes to U+FFFF in context without dropping or mutating markup
    uffff_parser = _break_before('\uffff', 'd')
    self.assertEqual(
      uffff_parser.translate_html_string('abc&#xffff;def', wrap=False),
      'abc\u200b&#xffff;\u200bdef',
    )
    # <plaintext>, <xmp>, and <noscript> raw text must not be split or double-escaped
    self.assertEqual(
      parser.translate_html_string(
        'abcdef<plaintext>abcdef<b>&amp;</b></plaintext>abcdef', wrap=False
      ),
      'abc\u200bdef<plaintext>abcdef<b>&amp;</b></plaintext>abcdef',
    )
    self.assertEqual(
      parser.translate_html_string(
        'abcdef<xmp>abcdef<b>&amp;</b></xmp>abcdef', wrap=False
      ),
      'abc\u200bdef<xmp>abcdef<b>&amp;</b></xmp>abc\u200bdef',
    )
    self.assertEqual(
      parser.translate_html_string(
        '<p>abcdef<noscript><p>abcdef&amp;</p></noscript>abcdef</p>', wrap=False
      ),
      '<p>abc\u200bdef<noscript><p>abcdef&amp;</p></noscript>abc\u200bdef</p>',
    )
    self.assertEqual(parser.translate_html_string('', wrap=False), '')
    self.assertEqual(
      parser.translate_html_string('', wrap=True),
      '<span style="word-break: keep-all; overflow-wrap: anywhere;"></span>',
    )

  def test_implicit_closing_and_unclosed_nobreak_containment(self) -> None:
    pair_parser = _break_between_pairs('cd')
    # Implicit <p> closed by <p> and <div>
    self.assertEqual(
      pair_parser.translate_html_string('<p>abc<p>def', wrap=False), '<p>abc<p>def'
    )
    self.assertEqual(
      pair_parser.translate_html_string('<p>abc<div>def</div>', wrap=False),
      '<p>abc<div>def</div>',
    )
    # Implicit <li> and <td>/<tr> closing isolates sibling items/cells
    self.assertEqual(
      pair_parser.translate_html_string('<ul><li>abc<li>def</ul>', wrap=False),
      '<ul><li>abc<li>def</ul>',
    )
    self.assertEqual(
      pair_parser.translate_html_string(
        '<table><tr><td>abc<td>def<tr><td>cd</table>', wrap=False
      ),
      '<table><tr><td>abc<td>def<tr><td>c\u200bd</table>',
    )
    # Unclosed <time> inside first <p> does not leak NO_BREAK into second <p>
    self.assertEqual(
      _break_before('d').translate_html_string(
        '<p><time>abcdef</p><p>abcdef</p>', wrap=False
      ),
      '<p><time>abcdef</p><p>abc\u200bdef</p>',
    )
