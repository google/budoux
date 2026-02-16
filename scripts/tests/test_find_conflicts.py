import os
import sys
import tempfile
import unittest

# module hack
LIB_PATH = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(LIB_PATH))

from scripts.find_conflicts import find_conflicts  # noqa (module hack)


class TestFindConflicts(unittest.TestCase):

  def setUp(self):
    self.temp_dir = tempfile.TemporaryDirectory()
    self.input_file = os.path.join(self.temp_dir.name, 'input.txt')
    self.output_file = os.path.join(self.temp_dir.name, 'output.txt')

  def tearDown(self):
    self.temp_dir.cleanup()

  def test_no_conflicts(self):
    # Setup data with no conflicts (different features)
    with open(self.input_file, 'w', encoding='utf-8') as f:
      f.write("1\tUW1:a\tUW2:b\n")
      f.write("-1\tUW1:c\tUW2:d\n")

    find_conflicts(self.input_file, self.output_file, strategy='delete_all')

    with open(self.output_file, 'r', encoding='utf-8') as f:
      lines = f.readlines()

    self.assertEqual(len(lines), 2)
    self.assertEqual(lines[0].strip(), "1\tUW1:a\tUW2:b")
    self.assertEqual(lines[1].strip(), "-1\tUW1:c\tUW2:d")

  def test_delete_all_conflicts(self):
    # Setup data with a conflict on UW1:a
    with open(self.input_file, 'w', encoding='utf-8') as f:
      f.write("1\tUW1:a\tUW2:b\n")  # Conflict
      f.write("-1\tUW1:a\tUW2:b\n")  # Conflict
      f.write("1\tUW1:c\tUW2:d\n")  # Safe

    find_conflicts(self.input_file, self.output_file, strategy='delete_all')

    with open(self.output_file, 'r', encoding='utf-8') as f:
      lines = f.readlines()

    self.assertEqual(len(lines), 1)
    self.assertEqual(lines[0].strip(), "1\tUW1:c\tUW2:d")

  def test_majority_conflict_above_threshold(self):
    # Setup data with 90% positive / 10% negative on UW1:x
    with open(self.input_file, 'w', encoding='utf-8') as f:
      for _ in range(9):
        f.write("5\tUW1:x\tUW2:y\n")
      for _ in range(1):
        f.write("-1\tUW1:x\tUW2:y\n")
      # Add a safe one
      f.write("1\tUW1:c\tUW2:d\n")

    # threshold 0.8 should keep the 9 positive entries
    find_conflicts(
        self.input_file, self.output_file, strategy='majority', threshold=0.8)

    with open(self.output_file, 'r', encoding='utf-8') as f:
      lines = f.readlines()

    self.assertEqual(len(lines), 10)
    # The 9 majority entries plus the 1 safe entry
    majority_count = sum(1 for line in lines if "5\tUW1:x\tUW2:y" in line)
    self.assertEqual(majority_count, 9)
    safe_count = sum(1 for line in lines if "1\tUW1:c\tUW2:d" in line)
    self.assertEqual(safe_count, 1)

  def test_majority_conflict_below_threshold(self):
    # Setup data with 60% positive / 40% negative on UW1:x
    with open(self.input_file, 'w', encoding='utf-8') as f:
      for _ in range(6):
        f.write("1\tUW1:x\tUW2:y\n")
      for _ in range(4):
        f.write("-1\tUW1:x\tUW2:y\n")
      # Add a safe one
      f.write("1\tUW1:c\tUW2:d\n")

    # threshold 0.8 means 0.6 is not enough, so it should default to delete_all
    find_conflicts(
        self.input_file, self.output_file, strategy='majority', threshold=0.8)

    with open(self.output_file, 'r', encoding='utf-8') as f:
      lines = f.readlines()

    # Only the safe one should survive
    self.assertEqual(len(lines), 1)
    self.assertEqual(lines[0].strip(), "1\tUW1:c\tUW2:d")

  def test_different_feature_order_is_same_conflict(self):
    # Setup data where features are same but different order
    with open(self.input_file, 'w', encoding='utf-8') as f:
      f.write("1\tUW1:a\tUW2:b\n")
      f.write("-1\tUW2:b\tUW1:a\n")

    find_conflicts(self.input_file, self.output_file, strategy='delete_all')

    with open(self.output_file, 'r', encoding='utf-8') as f:
      lines = f.readlines()

    self.assertEqual(len(lines), 0)


if __name__ == '__main__':
  unittest.main()
