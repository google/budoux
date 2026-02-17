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
import os
import sys
import tempfile
import unittest

# module hack
LIB_PATH = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(LIB_PATH))

from scripts.find_conflicts import find_conflicts  # noqa (module hack)


class TestFindConflicts(unittest.TestCase):

  def setUp(self) -> None:
    self.temp_dir = tempfile.TemporaryDirectory()
    self.input_file = os.path.join(self.temp_dir.name, 'input.txt')
    self.output_file = os.path.join(self.temp_dir.name, 'output.txt')

  def tearDown(self) -> None:
    self.temp_dir.cleanup()

  def test_no_conflicts(self) -> None:
    # Setup data with no conflicts (different features)
    with open(self.input_file, 'w', encoding='utf-8') as f:
      f.write("1\tUW1:a\tUW2:b\n")
      f.write("-1\tUW1:c\tUW2:d\n")

    find_conflicts(self.input_file, self.output_file, threshold=1.0)

    with open(self.output_file, 'r', encoding='utf-8') as f:
      lines = f.readlines()

    self.assertEqual(len(lines), 2)
    self.assertEqual(lines[0].strip(), "1\tUW1:a\tUW2:b")
    self.assertEqual(lines[1].strip(), "-1\tUW1:c\tUW2:d")

  def test_strict_threshold_deletes_all_conflicts(self) -> None:
    # Setup data with a conflict on UW1:a
    with open(self.input_file, 'w', encoding='utf-8') as f:
      f.write("1\tUW1:a\tUW2:b\n")  # Conflict 50%
      f.write("-1\tUW1:a\tUW2:b\n")  # Conflict 50%
      f.write("1\tUW1:c\tUW2:d\n")  # Safe 100%

    # Passing 1.0 acts as delete_all
    find_conflicts(self.input_file, self.output_file, threshold=1.0)

    with open(self.output_file, 'r', encoding='utf-8') as f:
      lines = f.readlines()

    self.assertEqual(len(lines), 1)
    self.assertEqual(lines[0].strip(), "1\tUW1:c\tUW2:d")

  def test_threshold_keeps_majority(self) -> None:
    # Setup data with 90% positive / 10% negative on UW1:x
    with open(self.input_file, 'w', encoding='utf-8') as f:
      for _ in range(9):
        f.write("5\tUW1:x\tUW2:y\n")
      for _ in range(1):
        f.write("-1\tUW1:x\tUW2:y\n")
      # Add a safe one
      f.write("1\tUW1:c\tUW2:d\n")

    # threshold 0.8 should keep the 9 positive entries
    find_conflicts(self.input_file, self.output_file, threshold=0.8)

    with open(self.output_file, 'r', encoding='utf-8') as f:
      lines = f.readlines()

    self.assertEqual(len(lines), 10)
    # The 9 majority entries plus the 1 safe entry
    majority_count = sum(1 for line in lines if "5\tUW1:x\tUW2:y" in line)
    self.assertEqual(majority_count, 9)
    safe_count = sum(1 for line in lines if "1\tUW1:c\tUW2:d" in line)
    self.assertEqual(safe_count, 1)

  def test_threshold_deletes_when_majority_not_met(self) -> None:
    # Setup data with 60% positive / 40% negative on UW1:x
    with open(self.input_file, 'w', encoding='utf-8') as f:
      for _ in range(6):
        f.write("1\tUW1:x\tUW2:y\n")
      for _ in range(4):
        f.write("-1\tUW1:x\tUW2:y\n")
      # Add a safe one
      f.write("1\tUW1:c\tUW2:d\n")

    # threshold 0.8 means 0.6 is not enough, so it should discard all conflicts
    find_conflicts(self.input_file, self.output_file, threshold=0.8)

    with open(self.output_file, 'r', encoding='utf-8') as f:
      lines = f.readlines()

    # Only the safe one should survive
    self.assertEqual(len(lines), 1)
    self.assertEqual(lines[0].strip(), "1\tUW1:c\tUW2:d")

  def test_different_feature_order_is_same_conflict(self) -> None:
    # Setup data where features are same but different order
    with open(self.input_file, 'w', encoding='utf-8') as f:
      f.write("1\tUW1:a\tUW2:b\n")
      f.write("-1\tUW2:b\tUW1:a\n")

    find_conflicts(self.input_file, self.output_file, threshold=1.0)

    with open(self.output_file, 'r', encoding='utf-8') as f:
      lines = f.readlines()

    self.assertEqual(len(lines), 0)

  def test_three_competing_labels(self) -> None:
    # Setup data with three competing labels
    with open(self.input_file, 'w', encoding='utf-8') as f:
      for _ in range(6):
        f.write("100\tUW1:z\n")  # 60%
      for _ in range(3):
        f.write("1\tUW1:z\n")  # 30%
      for _ in range(1):
        f.write("-1\tUW1:z\n")  # 10%

    find_conflicts(self.input_file, self.output_file, threshold=0.5)

    with open(self.output_file, 'r', encoding='utf-8') as f:
      lines = f.readlines()

    # Label "100" has 60% which > 0.5 threshold, so 6 items survive.
    self.assertEqual(len(lines), 6)
    self.assertEqual(lines[0].strip(), "100\tUW1:z")


if __name__ == '__main__':
  unittest.main()
