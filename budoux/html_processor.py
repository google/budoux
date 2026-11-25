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
"""HTML processor."""

import enum
import html
import json
import os
from html.parser import HTMLParser
from typing import Protocol

from ._html5_rules import (
  PLAINTEXT_INTERESTING,
  RAWTEXT_CLOSABLE_PATTERNS,
  build_implicit_close_rules,
)


class BoundaryParser(Protocol):
  """Protocol for parsers providing phrase boundary offsets."""

  def parse_boundaries(self, sentence: str) -> list[int]: ...


HTMLAttr = list[tuple[str, str | None]]
PARENT_CSS_STYLE = 'word-break: keep-all; overflow-wrap: anywhere;'
with open(
  os.path.join(os.path.dirname(__file__), 'html_spec.json'), encoding='utf-8'
) as f:
  HTML_SPEC = json.load(f)

SKIP_NODES: set[str] = set(HTML_SPEC['element_actions']['skip']) | set(
  HTML_SPEC['element_actions']['nobreak']
)
SPACE_CHARS: set[str] = set(HTML_SPEC['space_chars'])

# Elements without an end tag, see https://html.spec.whatwg.org/#void-elements
VOID_ELEMENTS: set[str] = {t.lower() for t in HTML_SPEC['void_elements']}

_BLOCK_NODES: frozenset[str] = frozenset(HTML_SPEC['element_actions']['block'])
_BLOCK_SKIP_NODES: frozenset[str] = frozenset(
  HTML_SPEC['element_actions']['block_skip']
)
_SKIP_ONLY_NODES: frozenset[str] = frozenset(HTML_SPEC['element_actions']['skip'])
_NO_BREAK_NODES: frozenset[str] = frozenset(HTML_SPEC['element_actions']['nobreak'])
_BREAK_NODES: frozenset[str] = frozenset(HTML_SPEC['element_actions']['break'])
_BREAK_OPPORTUNITY_NODES: frozenset[str] = frozenset(
  HTML_SPEC['element_actions']['break_opportunity']
)
_IMPLICIT_CLOSE_RULES = build_implicit_close_rules(_BLOCK_NODES, _BLOCK_SKIP_NODES)
_ZWSP = '\u200b'


class _DomAction(enum.Enum):
  INLINE = 0
  BLOCK = 1
  SKIP = 2
  NO_BREAK = 3
  BREAK = 4
  BREAK_OPPORTUNITY = 5


def _action_for_tag(tag_upper: str) -> _DomAction:
  if tag_upper in _SKIP_ONLY_NODES:
    return _DomAction.SKIP
  if tag_upper in _NO_BREAK_NODES:
    return _DomAction.NO_BREAK
  if tag_upper in _BREAK_NODES:
    return _DomAction.BREAK
  if tag_upper in _BREAK_OPPORTUNITY_NODES:
    return _DomAction.BREAK_OPPORTUNITY
  if tag_upper in _BLOCK_NODES:
    return _DomAction.BLOCK
  return _DomAction.INLINE


def _is_all_spaces(text: str) -> bool:
  return all(ch in SPACE_CHARS for ch in text)


class _TokenNode:
  """Base node in the lossless HTML token tree."""

  def serialize(self, separator: str) -> str:
    raise NotImplementedError


class _RawNode(_TokenNode):
  """Verbatim non-splittable HTML token (comment, decl, PI, CDATA, rawtext)."""

  def __init__(self, raw: str) -> None:
    self.raw = raw

  def serialize(self, separator: str) -> str:
    return self.raw


class _TextNode(_TokenNode):
  """Character data node mapping decoded context chars to verbatim HTML slices."""

  def __init__(self) -> None:
    self._chars: list[str] = []
    self.raw_pieces: list[str] = []
    self.internal_entity_offsets: set[int] = set()
    self.chunks: list[tuple[int, int]] = []
    self._cached_text: str | None = None

  @property
  def text(self) -> str:
    if self._cached_text is None:
      self._cached_text = ''.join(self._chars)
    return self._cached_text

  def append_plain(self, data: str) -> None:
    if not data:
      return
    self._cached_text = None
    for ch in data:
      self._chars.append(ch)
      self.raw_pieces.append(html.escape(ch, quote=False))

  def append_reference(self, decoded: str, raw: str) -> None:
    if not decoded:
      return
    self._cached_text = None
    base = len(self._chars)
    self._chars.append(decoded[0])
    self.raw_pieces.append(raw)
    for idx in range(1, len(decoded)):
      self._chars.append(decoded[idx])
      self.raw_pieces.append('')
      self.internal_entity_offsets.add(base + idx)

  def serialize(self, separator: str) -> str:
    if len(self.chunks) <= 1:
      return ''.join(self.raw_pieces)
    return separator.join(''.join(self.raw_pieces[s:e]) for s, e in self.chunks)


class _ElementNode(_TokenNode):
  """HTML element node preserving exact start and end tag markup."""

  def __init__(self, tag: str, start_tag_text: str) -> None:
    self.tag = tag
    self.tag_upper = tag.upper()
    self.start_tag_text = start_tag_text
    self.children: list[_TokenNode] = []
    self.end_tag_text = ''

  def serialize(self, separator: str) -> str:
    inner = ''.join(child.serialize(separator) for child in self.children)
    return f'{self.start_tag_text}{inner}{self.end_tag_text}'


class _TokenTreeBuilder(HTMLParser):
  """Single-pass lossless HTML tokenizer building a lightweight element tree."""

  @classmethod
  def parse(cls, source: str) -> _ElementNode:
    builder = cls(source)
    builder.feed(source)
    builder.close()
    return builder.root

  def __init__(self, source: str) -> None:
    super().__init__(convert_charrefs=False)
    self._source = source
    self._line_starts = [0]
    for idx, ch in enumerate(source):
      if ch == '\n':
        self._line_starts.append(idx + 1)
    self.root = _ElementNode('', '')
    self._stack: list[_ElementNode] = [self.root]

  def _pos(self) -> int:
    lineno, offset = self.getpos()
    return self._line_starts[lineno - 1] + offset

  def _current(self) -> _ElementNode:
    return self._stack[-1]

  def _append_text_node(self) -> _TextNode:
    children = self._current().children
    if children and isinstance(children[-1], _TextNode):
      return children[-1]
    node = _TextNode()
    children.append(node)
    return node

  def _pop_until_inclusive(self, target_index: int) -> None:
    while len(self._stack) > target_index:
      self._stack.pop()

  def _find_open_in_scope(
    self, targets: frozenset[str], boundaries: frozenset[str]
  ) -> int | None:
    for idx in range(len(self._stack) - 1, 0, -1):
      tag_upper = self._stack[idx].tag_upper
      if tag_upper in targets:
        return idx
      if tag_upper in boundaries:
        return None
    return None

  def _apply_implicit_close(self, new_tag_upper: str) -> None:
    for triggers, closable, boundaries in _IMPLICIT_CLOSE_RULES:
      if new_tag_upper in triggers:
        idx = self._find_open_in_scope(closable, boundaries)
        if idx is not None:
          self._pop_until_inclusive(idx)

  def _extract_raw_endtag(self, tag: str) -> str:
    pos = self._pos()
    n = len(self._source)
    i = pos + 2
    quote: str | None = None
    while i < n:
      ch = self._source[i]
      if quote is not None:
        if ch == quote:
          quote = None
      elif ch in ('"', "'"):
        quote = ch
      elif ch == '>':
        return self._source[pos : i + 1]
      i += 1
    return f'</{tag}>'

  def handle_starttag(self, tag: str, attrs: HTMLAttr) -> None:
    tag_upper = tag.upper()
    self._apply_implicit_close(tag_upper)
    raw_start = self.get_starttag_text() or f'<{tag}>'
    elem = _ElementNode(tag, raw_start)
    self._current().children.append(elem)
    if tag not in VOID_ELEMENTS:
      self._stack.append(elem)
    if tag == 'plaintext':
      self.cdata_elem = 'plaintext'
      self.interesting = PLAINTEXT_INTERESTING
    elif tag in RAWTEXT_CLOSABLE_PATTERNS:
      self.cdata_elem = tag
      self.interesting = RAWTEXT_CLOSABLE_PATTERNS[tag]

  def handle_startendtag(self, tag: str, attrs: HTMLAttr) -> None:
    tag_upper = tag.upper()
    self._apply_implicit_close(tag_upper)
    raw_start = self.get_starttag_text() or f'<{tag}/>'
    elem = _ElementNode(tag, raw_start)
    self._current().children.append(elem)

  def handle_endtag(self, tag: str) -> None:
    raw_end = self._extract_raw_endtag(tag)
    if tag in VOID_ELEMENTS:
      self._current().children.append(_RawNode(raw_end))
      return
    match_idx: int | None = None
    for idx in range(len(self._stack) - 1, 0, -1):
      if self._stack[idx].tag == tag:
        match_idx = idx
        break
    if match_idx is None:
      self._current().children.append(_RawNode(raw_end))
      return
    self._stack[match_idx].end_tag_text = raw_end
    self._pop_until_inclusive(match_idx)

  def handle_data(self, data: str) -> None:
    if not data:
      return
    if self.cdata_elem is not None:
      self._current().children.append(_RawNode(data))
      return
    self._append_text_node().append_plain(data)

  def _handle_reference(self, raw: str) -> None:
    if self.cdata_elem is not None:
      self._current().children.append(_RawNode(raw))
      return
    decoded = html.unescape(raw)
    if decoded and decoded != raw:
      self._append_text_node().append_reference(decoded, raw)
    elif not decoded and raw.startswith('&#'):
      # Python's html.unescape() strips Unicode noncharacters (e.g. &#xffff;) to '';
      # decode the code point for context while preserving the original reference.
      body = raw[2:].rstrip(';')
      codepoint = int(body[1:], 16) if body.startswith(('x', 'X')) else int(body, 10)
      self._append_text_node().append_reference(chr(codepoint), raw)
    else:
      self._append_text_node().append_plain(raw)

  def handle_entityref(self, name: str) -> None:
    pos = self._pos()
    end = pos + 1 + len(name)
    if end < len(self._source) and self._source[end] == ';':
      end += 1
    self._handle_reference(self._source[pos:end])

  def handle_charref(self, name: str) -> None:
    pos = self._pos()
    end = pos + 2 + len(name)
    if end < len(self._source) and self._source[end] == ';':
      end += 1
    self._handle_reference(self._source[pos:end])

  def handle_comment(self, data: str) -> None:
    pos = self._pos()
    if self._source.startswith('<!--', pos):
      close_idx = pos + 4 + len(data)
      if self._source.startswith('-->', close_idx):
        raw = f'<!--{data}-->'
      else:
        raw = self._source[pos:]
    else:
      gt_idx = self._source.find('>', pos)
      raw = self._source[pos : gt_idx + 1] if gt_idx >= 0 else self._source[pos:]
    self._current().children.append(_RawNode(raw))

  def handle_decl(self, decl: str) -> None:
    pos = self._pos()
    close_idx = pos + 2 + len(decl)
    if close_idx < len(self._source) and self._source[close_idx] == '>':
      raw = f'<!{decl}>'
    else:
      raw = self._source[pos:]
    self._current().children.append(_RawNode(raw))

  def handle_pi(self, data: str) -> None:
    pos = self._pos()
    close_idx = pos + 2 + len(data)
    if close_idx < len(self._source) and self._source[close_idx] == '>':
      raw = f'<?{data}>'
    else:
      raw = self._source[pos:]
    self._current().children.append(_RawNode(raw))

  def unknown_decl(self, data: str) -> None:
    pos = self._pos()
    close_idx = pos + 3 + len(data)
    if self._source.startswith(']]>', close_idx):
      raw = f'<![{data}]]>'
    elif self._source.startswith(']>', close_idx):
      raw = f'<![{data}]>'
    else:
      raw = self._source[pos:]
    self._current().children.append(_RawNode(raw))


class _NodeOrText:
  """Wraps either a splittable _TextNode or unbreakable NO_BREAK context text."""

  def __init__(
    self, text_node: _TextNode | None = None, unbreakable_text: str | None = None
  ) -> None:
    self.text_node = text_node
    self.unbreakable_text = unbreakable_text
    self.chunks: list[tuple[int, int]] = []
    self.has_break_opportunity_after = False

  @property
  def can_split(self) -> bool:
    return self.text_node is not None

  @property
  def text(self) -> str:
    if self.text_node is not None:
      return self.text_node.text
    return self.unbreakable_text or ''

  def commit_splits(self) -> None:
    if self.text_node is not None and len(self.chunks) > 1:
      self.text_node.chunks = self.chunks


class _Paragraph:
  """Represents an inline formatting unit collected for phrase segmentation."""

  def __init__(self) -> None:
    self.nodes: list[_NodeOrText] = []

  def is_empty(self) -> bool:
    return not self.nodes

  def has_splittable_node(self) -> bool:
    return any(node.can_split for node in self.nodes)

  def text(self) -> str:
    return ''.join(node.text for node in self.nodes)

  def mark_break_opportunity_after_last_node(self) -> None:
    if self.nodes:
      self.nodes[-1].has_break_opportunity_after = True

  def get_forced_opportunities(self) -> set[int]:
    opportunities: set[int] = set()
    offset = 0
    for node in self.nodes:
      t = node.text
      for i, ch in enumerate(t):
        if ch == _ZWSP:
          opportunities.add(offset + i)
          opportunities.add(offset + i + 1)
      if node.text_node is not None:
        for internal_off in node.text_node.internal_entity_offsets:
          opportunities.add(offset + internal_off)
      offset += len(t)
      if node.has_break_opportunity_after:
        opportunities.add(offset)
    return opportunities


class _ParagraphContext:
  """Mutable pointer to the active _Paragraph within a block formatting context."""

  def __init__(self) -> None:
    self.current = _Paragraph()

  def flush_into(self, out: list[_Paragraph]) -> None:
    if not self.current.is_empty():
      out.append(self.current)
      self.current = _Paragraph()


def _collect_from_element(
  element: _ElementNode,
  parent_context: _ParagraphContext | None,
  in_no_break: bool,
  out_paragraphs: list[_Paragraph],
) -> None:
  if element.tag_upper:
    action = _action_for_tag(element.tag_upper)
    if action == _DomAction.SKIP:
      if parent_context is not None and element.tag_upper in _BLOCK_SKIP_NODES:
        parent_context.flush_into(out_paragraphs)
      return
    if action == _DomAction.BREAK:
      if parent_context is not None and not parent_context.current.is_empty():
        parent_context.flush_into(out_paragraphs)
      return
    if action == _DomAction.BREAK_OPPORTUNITY:
      if parent_context is not None:
        parent_context.current.mark_break_opportunity_after_last_node()
      return
    is_new_block = parent_context is None or action == _DomAction.BLOCK
    child_no_break = in_no_break or action == _DomAction.NO_BREAK
  else:
    is_new_block = True
    child_no_break = in_no_break

  if is_new_block and parent_context is not None:
    parent_context.flush_into(out_paragraphs)

  ctx = _ParagraphContext() if is_new_block else parent_context
  assert ctx is not None

  for child in element.children:
    _collect_from_child_node(child, ctx, child_no_break, out_paragraphs)

  if is_new_block:
    ctx.flush_into(out_paragraphs)


def _collect_from_child_node(
  child: _TokenNode,
  ctx: _ParagraphContext,
  in_no_break: bool,
  out_paragraphs: list[_Paragraph],
) -> None:
  if isinstance(child, _ElementNode):
    _collect_from_element(child, ctx, in_no_break, out_paragraphs)
  elif isinstance(child, _TextNode):
    node_text = child.text
    if not node_text:
      return
    if in_no_break:
      ctx.current.nodes.append(_NodeOrText(unbreakable_text=node_text))
    else:
      ctx.current.nodes.append(_NodeOrText(text_node=child))


def _apply_to_paragraph(parser: BoundaryParser, paragraph: _Paragraph) -> None:
  if not paragraph.has_splittable_node():
    return
  text = paragraph.text()
  if not text or _is_all_spaces(text):
    return

  boundaries = parser.parse_boundaries(text)
  if not boundaries:
    return
  forced_opportunities = paragraph.get_forced_opportunities()

  adjusted_boundaries: list[int] = []
  text_len = len(text)
  for boundary in boundaries:
    if boundary <= 0 or boundary >= text_len:
      continue
    if boundary in forced_opportunities:
      continue
    if text[boundary - 1] in SPACE_CHARS or text[boundary] in SPACE_CHARS:
      continue
    adjusted_boundaries.append(boundary)

  if adjusted_boundaries:
    adjusted_boundaries.append(text_len + 1)
    _split_nodes(paragraph.nodes, adjusted_boundaries)


def _split_nodes(nodes: list[_NodeOrText], boundaries: list[int]) -> None:
  boundary_index = 0
  boundary = boundaries[0]
  node_start = 0
  last_splittable_node: _NodeOrText | None = None

  for node in nodes:
    node_text = node.text
    if not node_text:
      continue
    node_length = len(node_text)
    node_end = node_start + node_length

    if not node.can_split:
      if last_splittable_node is not None and boundary == node_start:
        last_len = len(last_splittable_node.text)
        if not last_splittable_node.chunks:
          last_splittable_node.chunks.append((0, last_len))
        last_splittable_node.chunks.append((last_len, last_len))
      while boundary < node_end:
        boundary_index += 1
        boundary = boundaries[boundary_index]
      last_splittable_node = None
      node_start = node_end
      continue

    last_splittable_node = node
    if boundary >= node_end:
      node_start = node_end
      continue

    chunk_start_in_node = 0
    while boundary < node_end:
      boundary_in_node = boundary - node_start
      node.chunks.append((chunk_start_in_node, boundary_in_node))
      chunk_start_in_node = boundary_in_node
      boundary_index += 1
      boundary = boundaries[boundary_index]
    node.chunks.append((chunk_start_in_node, node_length))
    node_start = node_end

  for node in nodes:
    node.commit_splits()


def translate(
  parser: BoundaryParser, html_str: str, separator: str = _ZWSP, wrap: bool = True
) -> str:
  """Translates an HTML string with phrase separators and optional span wrapping.

  Args:
    parser (BoundaryParser): The BudouX parser instance.
    html_str (str): The input HTML string.
    separator (str, optional): The separator string. Defaults to U+200B.
    wrap (bool, optional): Whether to wrap the output in a span element with
      inline styles. Defaults to True.

  Returns:
    The processed HTML string.
  """
  root = _TokenTreeBuilder.parse(html_str)

  paragraphs: list[_Paragraph] = []
  _collect_from_element(root, None, False, paragraphs)

  for paragraph in paragraphs:
    _apply_to_paragraph(parser, paragraph)

  resolved = root.serialize(separator)
  if not wrap:
    return resolved
  return f'<span style="{PARENT_CSS_STYLE}">{resolved}</span>'
