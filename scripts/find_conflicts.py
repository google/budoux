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


def _reconstruct_text(features: str) -> str:
  """Reconstructs the human-readable text from features."""
  feature_list = features.split('\t')
  unigrams = {}
  for feat in feature_list:
    if feat.startswith('UW'):
      parts = feat.split(':', 1)
      if len(parts) == 2:
        try:
          idx = int(parts[0][2:])
          unigrams[idx] = parts[1]
        except ValueError:
          pass

  # We expect UW1 to UW6
  left = "".join([unigrams.get(i, "　") for i in range(1, 4)])
  right = "".join([unigrams.get(i, "　") for i in range(4, 7)])
  return f"{left} / {right}"


def find_conflicts(data_path: str,
                   resolve_path: str = None,
                   strategy: str = "delete_all",
                   threshold: float = 0.8) -> None:
  """Finds and prints conflicting entries in the encoded data file.

    Args:
      data_path: The path to the encoded data file.
      resolve_path: The path to save the cleaned encoded data file.
      strategy: The resolution strategy ("delete_all" or "majority").
      threshold: The threshold for majority vote resolution.
    """

  features_to_labels: Dict[str, Set[int]] = defaultdict(set)
  features_to_count: Dict[str,
                          Dict[int,
                               int]] = defaultdict(lambda: defaultdict(int))
  total_data_points = 0

  # First pass: identify conflicts
  with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
      cols = line.strip('\n').split('\t')
      if len(cols) < 2:
        continue
      try:
        label = int(cols[0])
      except ValueError:
        continue

      # Canonicalize features by sorting them
      features = '\t'.join(sorted(cols[1:]))
      features_to_labels[features].add(label)
      features_to_count[features][label] += 1
      total_data_points += 1

  conflicts = {
      feat: labels
      for feat, labels in features_to_labels.items()
      if len(labels) > 1
  }

  deleted_points = 0
  resolved_features = set()
  majority_features = {}

  if conflicts:
    print(
        f"Found {len(conflicts)} unique feature sets with conflicting labels:\n"
    )
    for features, labels in conflicts.items():
      print(f"Features: {_reconstruct_text(features)}")
      total_for_feat = sum(
          features_to_count[features][label] for label in labels)
      winner_label = None
      max_ratio = 0.0

      for label in sorted(labels):
        count = features_to_count[features][label]
        ratio = count / total_for_feat
        print(f"  Label {label:2}: {count} occurrences ({ratio:.1%})")
        if ratio > max_ratio:
          max_ratio = ratio
          winner_label = label

      if strategy == "majority" and max_ratio >= threshold:
        majority_features[features] = winner_label
        print(f"  -> Resolved by majority vote: keeping label {winner_label}")
        # We delete all the ones that aren't the winner
        deleted_points += (
            total_for_feat - features_to_count[features][winner_label])
        resolved_features.add(features)
      else:
        print(f"  -> Conflicting: deleted all ({total_for_feat} occurrences)")
        deleted_points += total_for_feat
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
  if resolve_path:
    with open(
        data_path, 'r', encoding='utf-8') as fin, open(
            resolve_path, 'w', encoding='utf-8') as fout:
      for line in fin:
        cols = line.strip('\n').split('\t')
        if len(cols) < 2:
          continue
        try:
          label = int(cols[0])
        except ValueError:
          continue

        features = '\t'.join(sorted(cols[1:]))
        if features in resolved_features:
          if strategy == "majority" and features in majority_features:
            if label == majority_features[features]:
              fout.write(line)
          # if delete_all or didn't meet threshold, discard
        else:
          fout.write(line)

    print(f"Cleaned data saved to {resolve_path}")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('encoded_data', help='File path for the encoded data.')
  parser.add_argument(
      '-r',
      '--resolve',
      help='File path to save the cleaned encoded data.',
      default=None)
  parser.add_argument(
      '-s',
      '--strategy',
      choices=['delete_all', 'majority'],
      default='delete_all',
      help='Conflict resolution strategy (default: delete_all).')
  parser.add_argument(
      '-t',
      '--threshold',
      type=float,
      default=0.8,
      help='Threshold for majority vote (default: 0.8).')
  args = parser.parse_args()

  find_conflicts(args.encoded_data, args.resolve, args.strategy, args.threshold)


if __name__ == '__main__':
  main()
