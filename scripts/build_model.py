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
"""Builds a model from the learned weights.

This script outputs a model file in JSON format from the learned weights file
output by the `train.py` script.
"""

import argparse
import json
import typing


def rollup(weights_filename: str,
           model_filename: str,
           scale: int = 1000) -> None:
  """Rolls up the weights and outputs a model in JSON with integer scores.

  Args:
    weights_filename (str): A file path for the input weights file.
    model_filename (str): A file path for the output model file.
    scale (int, optional): A scale factor for the output score.
  """
  decision_trees: typing.Dict[str, float] = dict()
  with open(weights_filename) as f:
    for row in f.readlines():
      row = row.strip()
      if not row:
        continue
      feature = row.split('\t')[0]
      score = float(row.split('\t')[1])
      decision_trees.setdefault(feature, 0)
      decision_trees[feature] += score
  with open(model_filename, 'w', encoding='utf-8') as f:
    decision_trees_intscore = dict((item[0], int(item[1] * scale))
                                   for item in decision_trees.items()
                                   if abs(int(item[1] * scale)) > 0)
    json.dump(
        decision_trees_intscore, f, ensure_ascii=False, separators=(',', ':'))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      'weight_file', help='A file path for the learned weights.')
  parser.add_argument(
      '-o',
      '--outfile',
      help='A file path to export a model file. (default: model.json)',
      default='model.json')
  args = parser.parse_args()
  weights_filename = args.weight_file
  model_filename = args.outfile
  rollup(weights_filename, model_filename)
  print('Model file is exported as', model_filename)


if __name__ == '__main__':
  main()
