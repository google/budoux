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


def aggregate_scores(weights: list[str]) -> dict[str, dict[str, float]]:
  """Exports the model by aggregating the weight scores.

  Args:
    weights (List[str]): The lines of exported weight score file.

  Returns:
    model (Dict[string, Dict[string, float]]) The exported model.
  """
  decision_trees: dict[str, dict[str, float]] = {}
  for row in weights:
    row = row.strip()
    if not row:
      continue
    feature = row.split('\t')[0]
    feature_group, feature_content = feature.split(':', 1)
    score = float(row.split('\t')[1])
    decision_trees.setdefault(feature_group, {})
    decision_trees[feature_group].setdefault(feature_content, 0)
    decision_trees[feature_group][feature_content] += score
  return decision_trees


def round_model(
  model: dict[str, dict[str, float]], scale: int
) -> dict[str, dict[str, int]]:
  """Rounds the scores in the model to integer after scaling.

  Args:
    model (Dict[str, Dict[str, float]]): The model to round scores.
    scale (int, optional): A scale factor to multiply scores.

  Returns:
    model_rounded (Dict[str, Dict[str, int]]) The rounded model.
  """
  model_rounded: dict[str, dict[str, int]] = {}
  for feature_group, features in model.items():
    for feature_content, score in features.items():
      scaled_score = int(score * scale)
      if abs(scaled_score) > 0:
        model_rounded.setdefault(feature_group, {})
        model_rounded[feature_group][feature_content] = scaled_score
  return model_rounded


def parse_args(test: list[str] | None = None) -> argparse.Namespace:
  """Parses commandline arguments.

  Args:
    test (typing.Optional[typing.List[str]], optional): Commandline args for
      testing. Defaults to None.

  Returns:
    Parsed arguments (argparse.Namespace).
  """
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawTextHelpFormatter
  )
  parser.add_argument(
    'weight_file', help='A file path for the learned weights.'
  )
  parser.add_argument(
    '-o',
    '--outfile',
    help='A file path to export a model file. (default: model.json)',
    default='model.json',
    type=str,
  )
  parser.add_argument(
    '--scale',
    help='A scale factor for the output scores',
    default=1000,
    type=int,
  )
  if test is None:
    return parser.parse_args()
  else:
    return parser.parse_args(test)


def main() -> None:
  args = parse_args()
  weights_filename = args.weight_file
  model_filename = args.outfile
  scale = args.scale
  with open(weights_filename) as f:
    weights = f.readlines()
  model = aggregate_scores(weights)
  model_rounded = round_model(model, scale)
  with open(model_filename, 'w', encoding='utf-8') as f:
    json.dump(model_rounded, f, ensure_ascii=False, separators=(',', ':'))
  print('Model file is exported as', model_filename)


if __name__ == '__main__':
  main()
