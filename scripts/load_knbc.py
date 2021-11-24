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
"""Loads the KNBC corpus to generate training data."""

import argparse
import os
import tarfile
import typing
import urllib.request
import urllib.error
from html.parser import HTMLParser
from context import utils

RESOURCE_URL = (
    'https://nlp.ist.i.kyoto-u.ac.jp/kuntt/KNBC_v1.0_090925_utf8.tar.bz2')


class KNBCHTMLParser(HTMLParser):
    """Parses the HTML files in the KNBC corpus and outputs the chunks."""
    def __init__(self, split_tab: bool = True):
        super().__init__()
        self.chunks = ['']
        self.n_rows = 0
        self.n_cols = 0
        self.current_word = None
        self.split_tab = split_tab

    def handle_starttag(self, tag, _):
        if tag == 'tr':
            self.n_rows += 1
            self.n_cols = 0
            self.current_word = None
        if tag == 'td':
            self.n_cols += 1

    def handle_endtag(self, tag):
        if tag != 'tr': return
        if (self.n_rows > 2 and self.n_cols == 1
                and (self.split_tab or self.current_word == '文節区切り')):
            self.chunks.append('')
        if self.n_cols == 5:
            self.chunks[-1] += self.current_word

    def handle_data(self, data):
        if self.n_cols == 1:
            self.current_word = data


def break_before_open_parentheses(chunks: typing.List[str]):
    """Adds chunk breaks before every open parentheses.

  Args:
    chunks (List[str]): Source chunks.

  Returns:
    Processed chunks.
  """
    out: typing.List[str] = []
    for chunk in chunks:
        if '（' in chunk:
            index = chunk.index('（')
            if index > 0: out.append(chunk[:index])
            out.append(chunk[index:])
        else:
            out.append(chunk)
    return out


def postprocess(chunks: typing.List[str]):
    """Applies some processes to modify the extracted chunks.

  Args:
    chunks (List[str]): Source chunks.

  Returns:
    Processed chunks.
  """
    chunks = break_before_open_parentheses(chunks)
    return chunks


def download_knbc(target_dir: str):
    """Downloads the KNBC corpus and extracts files.

  Args:
    target_dir: A path to the directory to expand files.
  """
    os.makedirs(target_dir, exist_ok=True)
    download_file_path = os.path.join(target_dir, 'knbc.tar.bz2')
    try:
        urllib.request.urlretrieve(RESOURCE_URL, download_file_path)
    except urllib.error.HTTPError:
        print(f'\033[91mResource unavailable: {RESOURCE_URL}\033[0m')
        raise
    with tarfile.open(download_file_path, 'r:bz2') as t:
        t.extractall(path=target_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-o',
                        '--outfile',
                        help='''File path to output the training data.
            (default: source.txt)''',
                        default='source.txt')
    args = parser.parse_args()
    outfile = args.outfile
    html_dir = 'data/KNBC_v1.0_090925_utf8/html/'
    if not os.path.isdir(html_dir):
        download_knbc('data')
    with open(outfile, 'w') as f:
        for file in sorted(os.listdir(html_dir)):
            if file[-11:] != '-morph.html': continue
            parser = KNBCHTMLParser(split_tab=False)
            data = open(os.path.join(html_dir, file)).read()
            parser.feed(data)
            chunks = parser.chunks
            chunks = postprocess(chunks)
            if len(chunks) < 2: continue
            f.write(utils.SEP.join(chunks) + '\n')
    print('\033[92mTraining data is output to: %s\033[0m' % (outfile))


if __name__ == '__main__':
    main()
