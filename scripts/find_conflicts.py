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
"""Examines an encoded data file to find conflicting entries.

Conflicting entries are defined as those that have the same set of features
but different labels.
"""

import argparse
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def _reconstruct_text_from_unigram(features: str) -> str:
  """Reconstructs the human-readable text from unigrams."""
  feature_list = features.split('\t')
  unigrams = {}
  for feat in feature_list:
    if not feat.startswith('UW'):
      continue
    parts = feat.split(':', 1)
    if not len(parts) == 2:
      continue
    idx = int(parts[0][2:])
    unigrams[idx] = parts[1]

  # We expect UW1 to UW6
  full_width_space = chr(0x3000)
  left = "".join([unigrams.get(i, full_width_space) for i in range(1, 4)])
  right = "".join([unigrams.get(i, full_width_space) for i in range(4, 7)])
  return f"{left} / {right}"


def find_conflicts(data_path: str,
                   output_path: str,
                   threshold: float = 1.0) -> None:
  """Finds and prints conflicting entries in the encoded data file.

    Args:
      data_path: The path to the encoded data file.
      output_path: The path to save the cleaned encoded data file.
      threshold: The minimum ratio to keep the majority label (default 1.0 = unanimity).
    """

  features_to_pos_weight: Dict[str, int] = defaultdict(int)
  features_to_neg_weight: Dict[str, int] = defaultdict(int)
  features_to_pos_count: Dict[str, int] = defaultdict(int)
  features_to_neg_count: Dict[str, int] = defaultdict(int)
  total_data_points = 0

  # First pass: identify conflicts
  with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
      cols = line.strip('\n').split('\t')
      if len(cols) < 2:
        continue
      label = int(cols[0])

      # Canonicalize features by sorting them
      features = '\t'.join(sorted(cols[1:]))

      if label > 0:
        features_to_pos_weight[features] += label
        features_to_pos_count[features] += 1
      elif label < 0:
        features_to_neg_weight[features] += abs(label)
        features_to_neg_count[features] += 1

      total_data_points += 1

  # A conflict requires both positive and negative evidence
  conflicts = {
      feat for feat, pos_w in features_to_pos_weight.items()
      if pos_w > 0 and features_to_neg_weight[feat] > 0
  }

  deleted_points = 0
  resolved_features = set()
  majority_features = {}

  if conflicts:
    print(
        f"Found {len(conflicts)} unique feature sets with conflicting labels:\n"
    )
    for features in conflicts:
      print(f"Features: {_reconstruct_text_from_unigram(features)}")
      pos_w = features_to_pos_weight[features]
      neg_w = features_to_neg_weight[features]
      total_w = pos_w + neg_w
      pos_ratio = pos_w / total_w
      neg_ratio = neg_w / total_w

      print(
          f"  Positive weight: {pos_w} ({pos_ratio:.1%}) from {features_to_pos_count[features]} occurrences"
      )
      print(
          f"  Negative weight: {neg_w} ({neg_ratio:.1%}) from {features_to_neg_count[features]} occurrences"
      )

      if pos_ratio >= neg_ratio:
        max_ratio = pos_ratio
        winner_sign = 1
        winner_name = "positive"
        loser_count = features_to_neg_count[features]
      else:
        max_ratio = neg_ratio
        winner_sign = -1
        winner_name = "negative"
        loser_count = features_to_pos_count[features]

      if max_ratio >= threshold:
        majority_features[features] = winner_sign
        print(
            f"  -> Threshold met ({max_ratio:.1%} >= {threshold:.1%} limit): keeping {winner_name} sign"
        )
        # We delete all the ones that aren't the winner
        deleted_points += loser_count
        resolved_features.add(features)
      else:
        print(
            f"  -> Threshold not met: deleted all ({features_to_pos_count[features] + features_to_neg_count[features]} occurrences)"
        )
        deleted_points += features_to_pos_count[
            features] + features_to_neg_count[features]
        resolved_features.add(features)
      print("-" * 40)
  else:
    print("No conflicts found.")

  if deleted_points > 0:
    percent = (deleted_points / total_data_points) * 100
    print(
        f"Deleted {deleted_points} data points ({percent:.2f}% of {total_data_points} total)."
    )

  # Second pass: write out resolved file
  with open(
      data_path, 'r', encoding='utf-8') as fin, open(
          output_path, 'w', encoding='utf-8') as fout:
    for line in fin:
      cols = line.strip('\n').split('\t')
      label = int(cols[0])
      features = '\t'.join(sorted(cols[1:]))
      if features in resolved_features:
        if features in majority_features:
          winner_sign = majority_features[features]
          if (label > 0 and winner_sign > 0) or (label < 0 and winner_sign < 0):
            fout.write(line)
        # if threshold not met, discard
      else:
        fout.write(line)

    print(f"Cleaned data saved to {output_path}")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('encoded_data', help='File path for the encoded data.')
  parser.add_argument(
      '-o',
      '--output',
      help='File path to save the cleaned encoded data.',
      default=None)
  parser.add_argument(
      '-t',
      '--threshold',
      type=float,
      default=1.0,
      help='Threshold ratio for majority vote (default: 1.0 [Delete All]).')
  args = parser.parse_args()

  find_conflicts(args.encoded_data, args.output, args.threshold)


if __name__ == '__main__':
  main()
