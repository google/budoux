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
"""Compiled model evaluation benchmark utility.
"""

import argparse
import json
import os
import sys
import typing

# module hack to allow importing budoux from parent directory
LIB_PATH = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(LIB_PATH))

import budoux  # noqa: E402

EPS = 1e-9


def evaluate(model_path: str, test_data_path: str) -> typing.Dict[str, float]:
  """Loads the JSON model and evaluates it against the test dataset.

  Args:
    model_path (str): Path to the compiled model JSON file.
    test_data_path (str): Path to the test dataset file. Each line must contain
      one sentence split by '▁' (U+2581). For '.tsv' files, comment lines
      starting with '#' are ignored, and only the last column after tab
      splitting is evaluated.

  Returns:
    Dict[str, float]: A dictionary containing precision, recall, accuracy,
      and fscore.
  """
  with open(model_path, encoding='utf-8') as f:
    model = json.load(f)
  parser = budoux.Parser(model)

  tp = 0
  tn = 0
  fp = 0
  fn = 0

  is_tsv = test_data_path.endswith('.tsv')
  with open(test_data_path, encoding='utf-8') as f:
    for line in f:
      line = line.strip()
      if not line or line.startswith('#'):
        continue
      if is_tsv or '\t' in line:
        parts = line.split('\t')
        if len(parts) >= 2:
          line = parts[-1].strip()
        if not line:
          continue

      # Parse raw characters and ground truth break positions

      raw_chars: typing.List[str] = []
      ground_truth_breaks: typing.List[bool] = []
      next_is_break = False
      for char in line:
        if char == budoux.utils.SEP:
          next_is_break = True
        else:
          if raw_chars:
            ground_truth_breaks.append(next_is_break)
          next_is_break = False
          raw_chars.append(char)

      if not raw_chars:
        continue

      raw_sentence = ''.join(raw_chars)
      predicted_chunks = parser.parse(raw_sentence)

      # Build set of chunk start character indices in raw_sentence
      chunk_start_indices = set()
      curr_idx = 0
      for chunk in predicted_chunks:
        chunk_start_indices.add(curr_idx)
        curr_idx += len(chunk)

      # Extract predicted breaks (excluding starting character at index 0)
      predicted_breaks = []
      for i in range(1, len(raw_chars)):
        predicted_breaks.append(i in chunk_start_indices)

      # Accumulate evaluation counts across transitions
      for p, g in zip(predicted_breaks, ground_truth_breaks):
        if p and g:
          tp += 1
        elif not p and not g:
          tn += 1
        elif p and not g:
          fp += 1
        elif not p and g:
          fn += 1

  accuracy = (tp + tn) / (tp + tn + fp + fn + EPS)
  precision = tp / (tp + fp + EPS)
  recall = tp / (tp + fn + EPS)
  fscore = 2 * precision * recall / (precision + recall + EPS)

  return {
      'accuracy': accuracy,
      'precision': precision,
      'recall': recall,
      'fscore': fscore,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '-m',
      '--model',
      required=True,
      help='Path to the compiled model JSON file.')
  parser.add_argument(
      '-t',
      '--test-data',
      required=True,
      help=('Path to the test dataset file. Each line must contain one '
            'sentence split by "▁". For .tsv files, lines starting with "#" '
            'are ignored and only the last column after tab splitting is '
            'evaluated.'),
  )
  args = parser.parse_args()

  metrics = evaluate(args.model, args.test_data)
  print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
  main()
