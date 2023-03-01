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
"""Prepares a dataset from the KNBC corpus.

Before running this script, you need to download the KNBC corpus by running:

$ curl -o knbc.tar.bz2 https://nlp.ist.i.kyoto-u.ac.jp/kuntt/KNBC_v1.0_090925_utf8.tar.bz2
$ tar -xf knbc.tar.bz2

Now you should have a directory named `KNBC_v1.0_090925_utf8`.
Run the following to generate a dataset named `source_knbc.txt`.

$ python scripts/prepare_knbc.py KNBC_v1.0_090925_utf8 -o source_knbc.txt
"""

import argparse
import os
import typing
from html.parser import HTMLParser

from budoux import utils


class KNBCHTMLParser(HTMLParser):
  """Parses the HTML files in the KNBC corpus and outputs the chunks."""

  def __init__(self, split_tab: bool = True) -> None:
    super().__init__()
    self.chunks = ['']
    self.n_rows = 0
    self.n_cols = 0
    self.current_word: typing.Optional[str] = None
    self.split_tab = split_tab

  def handle_starttag(self, tag: str, _: typing.Any) -> None:
    if tag == 'tr':
      self.n_rows += 1
      self.n_cols = 0
      self.current_word = None
    if tag == 'td':
      self.n_cols += 1

  def handle_endtag(self, tag: str) -> None:
    if tag != 'tr':
      return None
    flag1 = self.n_rows > 2 and self.n_cols == 1
    flag2 = self.split_tab or self.current_word == '文節区切り'
    if flag1 and flag2:
      self.chunks.append('')
    if self.n_cols == 5 and type(self.current_word) is str:
      self.chunks[-1] += self.current_word

  def handle_data(self, data: str) -> None:
    if self.n_cols == 1:
      self.current_word = data


def break_before_sequence(chunks: typing.List[str],
                          sequence: str) -> typing.List[str]:
  """Breaks chunks before a specified character sequence appears.

  Args:
    chunks (List[str]): Chunks to break.
    sequence (str): A character sequence to break chunks before.

  Returns:
    Processed chunks.
  """
  chunks = utils.SEP.join(chunks).replace(sequence,
                                          utils.SEP + sequence).split(utils.SEP)
  chunks = [chunk for chunk in chunks if len(chunk) > 0]
  return chunks


def postprocess(chunks: typing.List[str]) -> typing.List[str]:
  """Applies some processes to modify the extracted chunks.

  Args:
    chunks (List[str]): Source chunks.

  Returns:
    Processed chunks.
  """
  chunks = break_before_sequence(chunks, '（')
  chunks = break_before_sequence(chunks, 'もら')
  return chunks


def parse_args() -> argparse.Namespace:
  DEFAULT_OUT_PATH = 'source.txt'
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument('source_dir', help='Path to the KNBC corpus directory.')
  parser.add_argument(
      '-o',
      '--outfile',
      help=f'File path to the output dataset. (default: {DEFAULT_OUT_PATH})',
      default=DEFAULT_OUT_PATH)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  source_dir = args.source_dir
  outfile = args.outfile
  html_dir = os.path.join(source_dir, 'html')
  with open(outfile, 'w') as f:
    for file in sorted(os.listdir(html_dir)):
      if file[-11:] != '-morph.html':
        continue
      parser = KNBCHTMLParser(split_tab=False)
      data = open(os.path.join(html_dir, file)).read()
      parser.feed(data)
      chunks = parser.chunks
      chunks = postprocess(chunks)
      if len(chunks) < 2:
        continue
      f.write(utils.SEP.join(chunks) + '\n')
  print('\033[92mTraining data is output to: %s\033[0m' % (outfile))


if __name__ == '__main__':
  main()
