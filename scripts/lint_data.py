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
"""Linter for BudouX training and fine-tuning datasets.

Validates that training datasets have no formatting syntax errors, duplicate
entries, or mutually contradictory feature labels.
"""

import argparse
import os
import sys
from collections import defaultdict

LIB_PATH = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(LIB_PATH))

from budoux import utils
from scripts import encode_data


def reconstruct_context(features: tuple[str, ...]) -> str:
  """Reconstructs readable 6-character context from unigram features."""
  unigrams: dict[int, str] = {}
  for feat in features:
    if feat.startswith("UW"):
      parts = feat.split(":", 1)
      if len(parts) == 2 and parts[0][2:].isdigit():
        unigrams[int(parts[0][2:])] = parts[1]
  left = "".join(unigrams.get(i, " ") for i in range(1, 4))
  right = "".join(unigrams.get(i, " ") for i in range(4, 7))
  return f"{left} | {right}"


def collect_files(paths: list[str]) -> list[str]:
  """Expands directory and file paths into a sorted list of text files."""
  files: list[str] = []
  for path in paths:
    if os.path.isdir(path):
      for root, _, filenames in os.walk(path):
        for fname in filenames:
          if fname.endswith(".txt"):
            files.append(os.path.join(root, fname))
    elif os.path.isfile(path):
      files.append(path)
    else:
      raise FileNotFoundError(f"Path not found: {path}")
  return sorted(set(files))


def lint_dataset(paths: list[str], check_conflicts: bool = True) -> list[str]:
  """Lints training dataset files for format errors and feature contradictions.

  Args:
    paths: List of file or directory paths to lint.
    check_conflicts: Whether to perform n-gram feature contradiction checks.

  Returns:
    A list of error message strings. Empty list indicates clean data.
  """
  errors: list[str] = []
  target_files = collect_files(paths)

  if not target_files:
    return ["No dataset files found to lint."]

  lines_seen: dict[str, tuple[str, int]] = {}
  feature_labels: dict[tuple[str, ...], dict[int, list[tuple[str, int, str, int]]]] = (
    defaultdict(lambda: defaultdict(list))
  )

  total_sentences = 0

  for fpath in target_files:
    try:
      with open(fpath, encoding="utf-8") as f:
        lines = f.readlines()
    except OSError as e:
      errors.append(f"{fpath}: Failed to read file: {e}")
      continue

    for line_num, raw_line in enumerate(lines, 1):
      line = raw_line.strip()
      if not line or line.startswith("#"):
        continue

      total_sentences += 1

      # 1. Format checks
      if utils.SEP + utils.SEP in line:
        errors.append(
          f"{fpath}:{line_num}: Consecutive separator markers detected: '{line}'"
        )
      if line.startswith(utils.SEP) or line.endswith(utils.SEP):
        errors.append(
          f"{fpath}:{line_num}: Leading or trailing separator marker detected: '{line}'"
        )
      if encode_data.INVALID in line:
        errors.append(
          f"{fpath}:{line_num}: Invalid marker character ('{encode_data.INVALID}') in source text: '{line}'"
        )

      # 2. Duplicate line checks
      if line in lines_seen:
        orig_file, orig_line = lines_seen[line]
        errors.append(
          f"{fpath}:{line_num}: Duplicate sentence (first seen at {orig_file}:{orig_line}): '{line}'"
        )
      else:
        lines_seen[line] = (fpath, line_num)

      # 3. Feature extraction & contradiction checks
      if check_conflicts:
        sentence, sep_indices = encode_data.normalize_input(line)
        for i in range(1, len(sentence) + 1):
          feat = tuple(
            sorted(
              encode_data.get_feature(
                sentence[i - 3] if i > 2 else encode_data.INVALID,
                sentence[i - 2] if i > 1 else encode_data.INVALID,
                sentence[i - 1],
                sentence[i] if i < len(sentence) else encode_data.INVALID,
                sentence[i + 1] if i + 1 < len(sentence) else encode_data.INVALID,
                sentence[i + 2] if i + 2 < len(sentence) else encode_data.INVALID,
              )
            )
          )
          is_break = i in sep_indices
          label = 1 if is_break else -1
          feature_labels[feat][label].append((fpath, line_num, line, i))

  if check_conflicts:
    conflicts = {
      feat: label_map
      for feat, label_map in feature_labels.items()
      if len(label_map) > 1
    }
    for feat, label_map in conflicts.items():
      context_str = reconstruct_context(feat)
      pos_instances = label_map[1]
      neg_instances = label_map[-1]

      msg = [f"Contradicting boundary decision for context [{context_str}]:"]
      msg.append(f"  Positive (break, {len(pos_instances)} occurrences):")
      for fpath, lnum, line, cidx in pos_instances:
        msg.append(f"    - {fpath}:{lnum} at char {cidx} in '{line}'")
      msg.append(f"  Negative (no break, {len(neg_instances)} occurrences):")
      for fpath, lnum, line, cidx in neg_instances:
        msg.append(f"    - {fpath}:{lnum} at char {cidx} in '{line}'")
      errors.append("\n".join(msg))

  return errors


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument(
    "paths",
    nargs="*",
    default=["data/finetuning"],
    help="Files or directories to lint (default: data/finetuning)",
  )
  parser.add_argument(
    "--format-only",
    action="store_true",
    help="Only perform syntax and duplicate checks, skip cross-sentence feature contradiction checks.",
  )

  args = parser.parse_args(argv)

  errors = lint_dataset(args.paths, check_conflicts=not args.format_only)
  if errors:
    print(f"\033[91m[FAIL]\033[0m Found {len(errors)} dataset issue(s):\n")
    for err in errors:
      print(f"  * {err}\n")
    return 1

  print(
    f"\033[92m[PASS]\033[0m All dataset checks passed successfully for {args.paths}."
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
