# Copyright 2026 Google LLC
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
"""Python-only HTML5 parsing rules for stdlib html.parser.HTMLParser.

Jsoup (Java) and browser/linkedom DOM parsers (JavaScript) already implement
HTML5 raw-text tokenization and optional end-tag closing natively, so these
minimal rules are kept out of the cross-language `html_spec.json`.
"""

import re

# Raw-text elements whose content must not be parsed as markup or split.
RAWTEXT_CLOSABLE_PATTERNS: dict[str, re.Pattern[str]] = {
  tag: re.compile(rf'</{tag}(?=[\t\n\r\f />])', re.IGNORECASE | re.ASCII)
  for tag in ('script', 'style', 'xmp', 'iframe', 'noembed', 'noframes', 'noscript')
}

# `<plaintext>` consumes all remaining input as raw text without a closing tag.
PLAINTEXT_INTERESTING: re.Pattern[str] = re.compile(r'\Z')

# Block-like elements in html_spec.json that do NOT implicitly close an open `<p>`.
_NON_P_CLOSING_BLOCKS: frozenset[str] = frozenset(
  {
    'BODY',
    'CAPTION',
    'COL',
    'DD',
    'DT',
    'HTML',
    'LEGEND',
    'LI',
    'MARQUEE',
    'TD',
    'TH',
    'TR',
  }
)

ImplicitCloseRule = tuple[frozenset[str], frozenset[str], frozenset[str]]


def build_implicit_close_rules(
  block_nodes: frozenset[str], block_skip_nodes: frozenset[str]
) -> tuple[ImplicitCloseRule, ...]:
  """Builds compact HTML5 optional end-tag rules: (triggers, closable, scope_boundaries)."""
  p_closing = (block_nodes - _NON_P_CLOSING_BLOCKS) | block_skip_nodes | {'HR'}
  return (
    (
      p_closing,
      frozenset({'P'}),
      frozenset({'BUTTON', 'HTML', 'TABLE', 'TD', 'TEMPLATE', 'TH'}),
    ),
    (
      frozenset({'LI'}),
      frozenset({'LI'}),
      frozenset({'DIR', 'HTML', 'MENU', 'OL', 'TABLE', 'TEMPLATE', 'UL'}),
    ),
    (
      frozenset({'DT', 'DD'}),
      frozenset({'DT', 'DD'}),
      frozenset({'DL', 'HTML', 'TABLE', 'TEMPLATE'}),
    ),
    (
      frozenset({'RT', 'RP'}),
      frozenset({'RT', 'RP'}),
      frozenset({'HTML', 'RUBY', 'TEMPLATE'}),
    ),
    (
      frozenset({'OPTION'}),
      frozenset({'OPTION'}),
      frozenset({'DATALIST', 'HTML', 'OPTGROUP', 'SELECT', 'TEMPLATE'}),
    ),
    (
      frozenset({'OPTGROUP'}),
      frozenset({'OPTION'}),
      frozenset({'DATALIST', 'HTML', 'SELECT', 'TEMPLATE'}),
    ),
    (
      frozenset({'OPTGROUP'}),
      frozenset({'OPTGROUP'}),
      frozenset({'DATALIST', 'HTML', 'SELECT', 'TEMPLATE'}),
    ),
    (
      frozenset({'TD', 'TH'}),
      frozenset({'TD', 'TH'}),
      frozenset({'HTML', 'TABLE', 'TEMPLATE', 'TR'}),
    ),
    (
      frozenset({'TR'}),
      frozenset({'TR', 'TD', 'TH'}),
      frozenset({'HTML', 'TABLE', 'TBODY', 'TEMPLATE', 'TFOOT', 'THEAD'}),
    ),
  )
