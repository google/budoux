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
      # Same sign, different weight is NOT a conflict
      f.write("5\tUW1:e\tUW2:f\n")
      f.write("2\tUW1:e\tUW2:f\n")

    find_conflicts(self.input_file, self.output_file, threshold=1.0)

    with open(self.output_file, 'r', encoding='utf-8') as f:
      lines = f.readlines()

    self.assertEqual(len(lines), 4)

  def test_strict_threshold_deletes_all_conflicts(self) -> None:
    # Conflict 50% pos vs 50% neg
    with open(self.input_file, 'w', encoding='utf-8') as f:
      f.write("1\tUW1:a\tUW2:b\n")  # Pos weight 1 (50%)
      f.write("-1\tUW1:a\tUW2:b\n")  # Neg weight 1 (50%)
      f.write("1\tUW1:c\tUW2:d\n")  # Safe 100%

    find_conflicts(self.input_file, self.output_file, threshold=1.0)

    with open(self.output_file, 'r', encoding='utf-8') as f:
      lines = f.readlines()

    self.assertEqual(len(lines), 1)
    self.assertEqual(lines[0].strip(), "1\tUW1:c\tUW2:d")

  def test_threshold_keeps_majority_summing_weights(self) -> None:
    # Setup data where weights sum to majority
    with open(self.input_file, 'w', encoding='utf-8') as f:
      f.write("5\tUW1:x\tUW2:y\n")  # Pos weight 5
      f.write("1\tUW1:x\tUW2:y\n")  # Pos weight 1. Total pos = 6
      f.write("-1\tUW1:x\tUW2:y\n")  # Neg weight 1. Total neg = 1
      f.write("1\tUW1:c\tUW2:d\n")  # Safe

    # threshold 0.8 -> 6/7 = ~85.7% > 80%
    find_conflicts(self.input_file, self.output_file, threshold=0.8)

    with open(self.output_file, 'r', encoding='utf-8') as f:
      lines = f.readlines()

    self.assertEqual(len(lines), 3)
    # The 2 positive entries plus the 1 safe entry
    majority_count = sum(1 for line in lines if "UW1:x\tUW2:y" in line)
    self.assertEqual(majority_count, 2)

  def test_threshold_deletes_when_majority_not_met(self) -> None:
    # Pos weight = 5, Neg weight = 4. 5/9 = 55.5%
    with open(self.input_file, 'w', encoding='utf-8') as f:
      f.write("5\tUW1:x\tUW2:y\n")
      f.write("-4\tUW1:x\tUW2:y\n")
      f.write("1\tUW1:c\tUW2:d\n")

    # threshold 0.8 > 55% -> discard all
    find_conflicts(self.input_file, self.output_file, threshold=0.8)

    with open(self.output_file, 'r', encoding='utf-8') as f:
      lines = f.readlines()

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


if __name__ == '__main__':
  unittest.main()
