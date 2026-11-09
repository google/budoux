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
"""Tests the training script."""

import math
import os
import tempfile
import unittest

import pytest

pytest.importorskip("jax")
from jax import numpy as jnp

from scripts import train


class TestArgParse(unittest.TestCase):
  def test_cmdargs_invalid_option(self) -> None:
    cmdargs = ['-v']
    with self.assertRaises(SystemExit) as cm:
      train.parse_args(cmdargs)
    self.assertEqual(cm.exception.code, 2)

  def test_cmdargs_help(self) -> None:
    cmdargs = ['-h']
    with self.assertRaises(SystemExit) as cm:
      train.parse_args(cmdargs)
    self.assertEqual(cm.exception.code, 0)

  def test_cmdargs_no_data(self) -> None:
    with self.assertRaises(SystemExit) as cm:
      train.parse_args([])
    self.assertEqual(cm.exception.code, 2)

  def test_cmdargs_default(self) -> None:
    cmdargs = ['encoded.txt']
    output = train.parse_args(cmdargs)
    self.assertEqual(output.encoded_train_data, 'encoded.txt')
    self.assertEqual(output.output, train.DEFAULT_OUTPUT_NAME)
    self.assertEqual(output.log, train.DEFAULT_LOG_NAME)
    self.assertEqual(output.feature_thres, train.DEFAULT_FEATURE_THRES)
    self.assertEqual(output.iter, train.DEFAULT_ITERATION)
    self.assertEqual(output.out_span, 1000)
    self.assertEqual(output.val_data, None)
    self.assertEqual(output.algorithm, train.DEFAULT_ALGORITHM)
    self.assertEqual(output.learning_rate, train.DEFAULT_LEARNING_RATE)

  def test_cmdargs_out_span_dynamic_default(self) -> None:
    output1 = train.parse_args(['encoded.txt', '--iter', '200000'])
    self.assertEqual(output1.out_span, 5000)
    output2 = train.parse_args(['encoded.txt', '--iter', '5'])
    self.assertEqual(output2.out_span, 1)

  def test_cmdargs_full(self) -> None:
    cmdargs = [
      'encoded.txt',
      '-o',
      'out.txt',
      '--log',
      'foo.log',
      '--feature-thres',
      '100',
      '--iter',
      '10',
      '--out-span',
      '50',
      '--val-data',
      'val_encoded.txt',
      '--algorithm',
      'logitboost',
      '--learning-rate',
      '0.5',
    ]
    output = train.parse_args(cmdargs)
    self.assertEqual(output.encoded_train_data, 'encoded.txt')
    self.assertEqual(output.output, 'out.txt')
    self.assertEqual(output.log, 'foo.log')
    self.assertEqual(output.feature_thres, 100)
    self.assertEqual(output.iter, 10)
    self.assertEqual(output.out_span, 50)
    self.assertEqual(output.val_data, 'val_encoded.txt')
    self.assertEqual(output.algorithm, 'logitboost')
    self.assertEqual(output.learning_rate, 0.5)
    self.assertEqual(output.patience, None)
    self.assertEqual(output.min_delta, 0.0001)

  def test_cmdargs_with_learning_rate(self) -> None:
    cmdargs = ['encoded.txt', '--learning-rate', '0.2']
    output = train.parse_args(cmdargs)
    self.assertEqual(output.learning_rate, 0.2)

    cmdargs_short = ['encoded.txt', '-lr', '0.1']
    output_short = train.parse_args(cmdargs_short)
    self.assertEqual(output_short.learning_rate, 0.1)

  def test_cmdargs_with_patience(self) -> None:
    cmdargs = ['encoded.txt', '--patience', '3', '--min-delta', '0.001']
    output = train.parse_args(cmdargs)
    self.assertEqual(output.patience, 3)
    self.assertEqual(output.min_delta, 0.001)

  def test_cmdargs_invalid_algorithm(self) -> None:
    cmdargs = ['encoded.txt', '--algorithm', 'invalid']
    with self.assertRaises(SystemExit) as cm:
      train.parse_args(cmdargs)
    self.assertEqual(cm.exception.code, 2)


class TestPreprocess(unittest.TestCase):
  def test_standard_setup(self) -> None:
    with tempfile.NamedTemporaryFile(delete=False) as tf:
      train_data_path = tf.name
      self.addCleanup(os.remove, tf.name)
    with tempfile.NamedTemporaryFile(delete=False) as tf:
      val_data_path = tf.name
      self.addCleanup(os.remove, tf.name)
    with open(train_data_path, 'w') as f:
      f.write('1\tfoo\tbar\n-1\tfoo\n1\tfoo\tbar\tbaz\n1\tbar\tfoo\n-1\tbaz\tqux\n')
    with open(val_data_path, 'w') as f:
      f.write('1\tbar\tbaz\n-1\txyz\n1\tabc\tqux\tfoo\n')
    train_dataset, features, val_dataset = train.preprocess(
      train_data_path, 1, val_data_path
    )

    self.assertEqual(features, ['foo', 'bar', 'baz'])
    # The training input X and the target Y should look like below:
    # Y    X(foo bar baz)
    # 1      1   1   0
    # -1     1   0   0
    # 1      1   1   1
    # 1      1   1   0
    # -1     0   0   1
    self.assertEqual(train_dataset.Y.tolist(), [1, -1, 1, 1, -1])
    self.assertEqual(train_dataset.X_rows.tolist(), [0, 0, 1, 2, 2, 2, 3, 3, 4])
    self.assertEqual(train_dataset.X_cols.tolist(), [0, 1, 0, 0, 1, 2, 1, 0, 2])

    # The validation input X and the target Y should look like below:
    # Y    X(foo bar baz)
    # 1      0   1   1
    # -1     0   0   0
    # 1      1   0   0
    self.assertIsInstance(val_dataset, train.Dataset)
    if isinstance(val_dataset, train.Dataset):
      self.assertEqual(val_dataset.Y.tolist(), [1, -1, 1])
      self.assertEqual(val_dataset.X_rows.tolist(), [0, 0, 2])
      self.assertEqual(val_dataset.X_cols.tolist(), [1, 2, 0])
    else:
      raise TypeError('val_dataset is not an instance of Dataset.')

  def test_no_val(self) -> None:
    with tempfile.NamedTemporaryFile(delete=False) as tf:
      train_data_path = tf.name
      self.addCleanup(os.remove, tf.name)
    with open(train_data_path, 'w') as f:
      f.write('1\tfoo\tbar\n-1\tfoo\n1\tfoo\tbar\tbaz\n1\tbar\tfoo\n-1\tbaz\tqux\n')
    train_dataset, features, val_dataset = train.preprocess(train_data_path, 1)
    self.assertEqual(features, ['foo', 'bar', 'baz'])
    self.assertEqual(train_dataset.Y.tolist(), [1, -1, 1, 1, -1])
    self.assertEqual(train_dataset.X_rows.tolist(), [0, 0, 1, 2, 2, 2, 3, 3, 4])
    self.assertEqual(train_dataset.X_cols.tolist(), [0, 1, 0, 0, 1, 2, 1, 0, 2])
    self.assertIsNone(val_dataset)


class TestPred(unittest.TestCase):
  def test_standard_setup(self) -> None:
    X = jnp.array([[1, 1, 0], [1, 0, 1], [0, 1, 0], [0, 0, 1]])
    phis = jnp.array([0.4, 0.2, -0.3])
    N = X.shape[0]
    rows, cols = jnp.where(X == 1)
    res = train.pred(phis, rows, cols, N)
    expected = [
      0.4 + 0.2 - (-0.3) > 0,
      0.4 - 0.2 + (-0.3) > 0,
      -0.4 + 0.2 - (-0.3) > 0,
      -0.4 - 0.2 + (-0.3) > 0,
    ]
    self.assertEqual(res.tolist(), expected)


class TestGetMetrics(unittest.TestCase):
  def test_standard_setup(self) -> None:
    pred = jnp.array([0, 0, 1, 0, 0], dtype=bool)
    target = jnp.array([1, 0, 1, 1, 1], dtype=bool)
    result = train.get_metrics(pred, target)
    self.assertEqual(result.tp, 1)
    self.assertEqual(result.tn, 1)
    self.assertEqual(result.fp, 0)
    self.assertEqual(result.fn, 3)
    self.assertEqual(result.accuracy, 2 / 5)
    p = 1 / 1
    r = 1 / 4
    self.assertEqual(result.precision, p)
    self.assertEqual(result.recall, r)
    self.assertEqual(result.fscore, 2 * p * r / (p + r))


class TestUpdate(unittest.TestCase):
  X = jnp.array([[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0]])

  def test_standard_setup1(self) -> None:
    rows, cols = jnp.where(self.X == 1)
    M = self.X.shape[-1]
    Y = jnp.array([1, 1, 0, 0, 1], dtype=bool)
    w = jnp.array([0.1, 0.3, 0.1, 0.1, 0.4])
    scores = jnp.zeros(M)
    new_w, new_scores, best_feature_index, added_score = train.update(
      w, scores, rows, cols, Y
    )
    self.assertFalse(w.argmax() == 0)
    self.assertTrue(new_w.argmax() == 0)
    self.assertFalse(scores.argmax() == 1)
    self.assertTrue(new_scores.argmax() == 1)
    self.assertEqual(best_feature_index, 1)
    self.assertTrue(added_score > 0)


class TestUpdateLogitBoost(unittest.TestCase):
  X = jnp.array([[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0]])

  def test_standard_setup(self) -> None:
    rows, cols = jnp.where(self.X == 1)
    M = self.X.shape[-1]
    N = self.X.shape[0]
    Y = jnp.array([1.0, 1.0, 0.0, 0.0, 1.0])
    F = jnp.zeros(N, dtype=jnp.float32)
    scores = jnp.zeros(M)
    new_F, new_scores, best_feature_index, added_score = train.update_logitboost(
      F, scores, rows, cols, M, N, Y
    )
    self.assertFalse(scores.argmax() == 1)
    self.assertTrue(new_scores.argmax() == 1)
    self.assertEqual(best_feature_index, 1)
    self.assertTrue(added_score > 0)
    self.assertFalse(jnp.all(new_F == 0))

  def test_sample_weights(self) -> None:
    rows, cols = jnp.where(self.X == 1)
    M = self.X.shape[-1]
    N = self.X.shape[0]
    Y = jnp.array([1.0, 1.0, 0.0, 0.0, 1.0])
    F = jnp.zeros(N, dtype=jnp.float32)
    scores = jnp.zeros(M)
    # Heavy weight on sample index 0 (feature 0 and 2 are active)
    weights = jnp.array([100.0, 1.0, 1.0, 1.0, 1.0])
    _, _, best_feature_index, _ = train.update_logitboost(
      F, scores, rows, cols, M, N, Y, sample_weights=weights
    )
    self.assertIn(best_feature_index, [0, 2])

  def test_learning_rate(self) -> None:
    rows, cols = jnp.where(self.X == 1)
    M = self.X.shape[-1]
    N = self.X.shape[0]
    Y = jnp.array([1.0, 1.0, 0.0, 0.0, 1.0])
    F = jnp.zeros(N, dtype=jnp.float32)
    scores = jnp.zeros(M)
    _, _, feat_1, score_1 = train.update_logitboost(
      F, scores, rows, cols, M, N, Y, learning_rate=1.0
    )
    _, _, feat_half, score_half = train.update_logitboost(
      F, scores, rows, cols, M, N, Y, learning_rate=0.5
    )
    self.assertEqual(feat_1, feat_half)
    self.assertAlmostEqual(score_half, score_1 * 0.5, places=5)

  def test_scale_invariance(self) -> None:
    rows, cols = jnp.where(self.X == 1)
    M = self.X.shape[-1]
    N = self.X.shape[0]
    Y = jnp.array([1.0, 1.0, 0.0, 0.0, 1.0])
    F = jnp.zeros(N, dtype=jnp.float32)
    scores = jnp.zeros(M)
    w_unit = jnp.ones(N)
    w_scaled = 5.0 * jnp.ones(N)
    _, _, feat_unit, score_unit = train.update_logitboost(
      F, scores, rows, cols, M, N, Y, sample_weights=w_unit
    )
    _, _, feat_scaled, score_scaled = train.update_logitboost(
      F, scores, rows, cols, M, N, Y, sample_weights=w_scaled
    )
    self.assertEqual(feat_unit, feat_scaled)
    self.assertAlmostEqual(score_unit, score_scaled, places=5)

  def test_one_sided_floor_on_dense_breaks(self) -> None:
    # 75% positive boundaries including an OOV positive row (unconstrained optimum
    # would have sum(scores) < 0, i.e., parser base_score > 0).
    X = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
    rows, cols = jnp.where(X == 1)
    M = 3
    N = X.shape[0]
    Y = jnp.array([1.0, 1.0, 0.0, 1.0])
    F = jnp.zeros(N, dtype=jnp.float32)
    scores = jnp.zeros(M)

    def logistic_loss(s: jnp.ndarray) -> float:
      margin = (2.0 * X.astype(jnp.float32) - 1.0) @ s
      return float(jnp.sum(jnp.logaddexp(0.0, -2.0 * (2.0 * Y - 1.0) * margin)))

    initial_loss = logistic_loss(scores)
    for _ in range(40):
      F, scores, _, _ = train.update_logitboost(F, scores, rows, cols, M, N, Y)

    self.assertGreaterEqual(float(jnp.sum(scores)), train.DEFAULT_MIN_SCORE_SUM - 1e-5)
    self.assertLess(logistic_loss(scores), initial_loss)

  def test_sparse_breaks_not_clamped_down(self) -> None:
    # Low break density (20% Y=1, including OOV Y=0 rows): sum(scores) must grow
    # freely well past 1.0 instead of being pulled down toward 0.5.
    X = jnp.array([[1, 0], [0, 1], [0, 0], [0, 0], [0, 0]])
    rows, cols = jnp.where(X == 1)
    M = 2
    N = X.shape[0]
    Y = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0])
    F = jnp.zeros(N, dtype=jnp.float32)
    scores = jnp.zeros(M)

    for _ in range(25):
      F, scores, _, _ = train.update_logitboost(F, scores, rows, cols, M, N, Y)

    self.assertGreater(float(jnp.sum(scores)), 1.2)

  def test_min_sum_none_disables_floor(self) -> None:
    # Dense breaks where unconstrained 1D Newton step at t=0 is negative (-1.0)
    # and unconstrained sum(scores) drifts well below 0.
    X = jnp.array([[0, 0], [0, 0], [0, 1]])
    rows, cols = jnp.where(X == 1)
    M = 2
    N = X.shape[0]
    Y = jnp.array([1.0, 1.0, 0.0])
    F = jnp.zeros(N, dtype=jnp.float32)
    scores = jnp.zeros(M)

    # At t=0: g = [0.5, 0.5, -0.5], sum(g) = 0.5, g_sparse = [0.0, -0.5],
    # G = [-0.5, -1.5], sum(h) = 0.75 -> exact Newton step on feature 1 is -1.0.
    F, scores, feat_idx, step_score = train.update_logitboost(
      F, scores, rows, cols, M, N, Y, min_sum=None
    )
    self.assertEqual(int(feat_idx), 1)
    self.assertAlmostEqual(float(step_score), -1.0, places=5)
    self.assertLess(float(jnp.sum(scores)), 0.0)

    for _ in range(10):
      F, scores, _, _ = train.update_logitboost(
        F, scores, rows, cols, M, N, Y, min_sum=None
      )
    self.assertLess(float(jnp.sum(scores)), -1.0)

  def test_mid_floor_avoids_step_size_starvation(self) -> None:
    # cur_sum = 0.58 lies between mid_floor (0.55) and upper_floor (0.60).
    # Unconstrained step is negative, triggering the floor handler; with the
    # midpoint threshold, need_pos is False so a negative step is taken rather
    # than clipping a positive step to upper_floor - cur_sum (+0.02).
    X = jnp.array([[1, 0], [0, 1], [0, 0], [0, 0]])
    rows, cols = jnp.where(X == 1)
    M = 2
    N = X.shape[0]
    Y = jnp.array([1.0, 0.0, 1.0, 1.0])
    scores = jnp.array([0.58, 0.0], dtype=jnp.float32)
    F = (2.0 * X.astype(jnp.float32) - 1.0) @ scores

    _, new_scores, best_idx, step_score = train.update_logitboost(
      F, scores, rows, cols, M, N, Y, learning_rate=1.0, min_sum=0.5
    )
    self.assertEqual(int(best_idx), 1)
    self.assertLess(float(step_score), 0.0)
    self.assertGreaterEqual(float(jnp.sum(new_scores)), 0.5 - 1e-6)

  def test_floor_holds_when_all_g_sparse_nonpositive(self) -> None:
    # Y is all zeros so g = -p < 0 everywhere (all g_sparse <= 0). Feature 0 is
    # active on all 3 rows, so G_0 = 2 * g_sparse[0] - sum(g) < 0 and
    # unconstrained_score < 0. Starting at cur_sum >= min_sum must never drop
    # below min_sum across both cur_sum >= mid_floor and cur_sum < mid_floor.
    X = jnp.array([[1, 1], [1, 0], [1, 0]])
    rows, cols = jnp.where(X == 1)
    M = 2
    N = X.shape[0]
    Y = jnp.array([0.0, 0.0, 0.0])
    scores = jnp.array([0.30, 0.28], dtype=jnp.float32)  # cur_sum = 0.58 >= 0.50
    F = (2.0 * X.astype(jnp.float32) - 1.0) @ scores

    for _ in range(10):
      F, scores, _, _ = train.update_logitboost(
        F, scores, rows, cols, M, N, Y, learning_rate=1.0, min_sum=0.5
      )
      self.assertGreaterEqual(float(jnp.sum(scores)), 0.5 - 1e-6)

  def test_floor_selects_by_second_order_gain(self) -> None:
    # Feature 0 has slightly larger 1st-order gradient (g_sparse[0] = 2.0 vs
    # g_sparse[1] ≈ 1.964), but much higher active curvature (h_sparse[0] = 4.0
    # vs h_sparse[1] ≈ 0.035), so Feature 1 has higher 2nd-order Taylor gain.
    X = jnp.array([[1, 0]] * 16 + [[0, 1]] * 2 + [[0, 0]] * 6, dtype=jnp.int32)
    rows, cols = jnp.where(X == 1)
    M = 2
    N = X.shape[0]
    Y = jnp.array([1.0] * 10 + [0.0] * 6 + [1.0] * 2 + [1.0] * 6, dtype=jnp.float32)
    scores = jnp.array([0.25, 0.25], dtype=jnp.float32)  # cur_sum = 0.50 < mid_floor
    F = jnp.array([0.0] * 16 + [-2.0] * 2 + [-2.0] * 6, dtype=jnp.float32)

    _, _, best_idx, step_score = train.update_logitboost(
      F, scores, rows, cols, M, N, Y, learning_rate=1.0, min_sum=0.5
    )
    self.assertEqual(int(best_idx), 1)
    self.assertGreater(float(step_score), 0.0)


class TestFit(unittest.TestCase):
  def test_fit(self) -> None:
    with tempfile.NamedTemporaryFile(delete=False) as tf:
      weights_file_path = tf.name
      self.addCleanup(os.remove, tf.name)
    with tempfile.NamedTemporaryFile(delete=False) as tf:
      log_file_path = tf.name
      self.addCleanup(os.remove, tf.name)
    # Prepare a dataset that the 2nd feature (= the 2nd col in X) perfectly
    # correlates with Y in a negative way.
    X = jnp.array([[0, 1, 1, 1], [1, 1, 0, 1], [0, 0, 1, 1], [1, 0, 0, 1]])
    Y = jnp.array([-1, -1, 1, 1])
    rows, cols = jnp.where(X == 1)
    dataset = train.Dataset(rows, cols, Y)
    features = ['a', 'b', 'c']
    iters = 5
    out_span = 2
    scores = train.fit(
      dataset, dataset, features, iters, weights_file_path, log_file_path, out_span
    )
    with open(weights_file_path) as f:
      weights = [line.split('\t') for line in f.read().splitlines() if line.strip()]
    top_feature = weights[0][0]
    self.assertEqual(
      top_feature, 'b', msg='The most effective feature should be selected.'
    )
    self.assertEqual(
      len(weights),
      iters,
      msg='The number of lines should equal to the iteration count.',
    )

    with open(log_file_path) as f:
      log = [line.split('\t') for line in f.read().splitlines() if line.strip()]
    self.assertEqual(
      log[0],
      [
        'iter',
        'train_accuracy',
        'train_precision',
        'train_recall',
        'train_fscore',
        'val_accuracy',
        'val_precision',
        'val_recall',
        'val_fscore',
      ],
    )
    self.assertEqual(
      len(log),
      math.ceil(iters / out_span) + 1,
      msg='The number of lines should equal to the ceil of iteration / out_span plus one for the header',
    )
    self.assertEqual(
      len({len(line) for line in log}),
      1,
      msg='The header and the body should have the same number of columns.',
    )

    model: dict[str, float] = {}
    for weight in weights:
      model.setdefault(weight[0], 0)
      model[weight[0]] += float(weight[1])
    self.assertEqual(scores.shape[0], len(features))
    loaded_scores = jnp.array([model.get(feature, 0) for feature in features])
    self.assertTrue(jnp.all(jnp.isclose(scores, loaded_scores)))

  def test_fit_with_early_stopping(self) -> None:
    dataset_train = train.Dataset(
      jnp.asarray([0, 0, 1, 2, 2, 3]),
      jnp.asarray([0, 1, 0, 0, 1, 1]),
      jnp.asarray([1, -1, 1, -1]),
    )
    dataset_val = train.Dataset(
      jnp.asarray([0, 1]), jnp.asarray([0, 1]), jnp.asarray([1, -1])
    )
    features = ['foo', 'bar']
    with (
      tempfile.NamedTemporaryFile(delete=False) as tf_w,
      tempfile.NamedTemporaryFile(delete=False) as tf_l,
    ):
      self.addCleanup(os.remove, tf_w.name)
      self.addCleanup(os.remove, tf_l.name)
      train.fit(
        dataset_train,
        dataset_val,
        features,
        iters=50,
        weights_filename=tf_w.name,
        log_filename=tf_l.name,
        out_span=2,
        patience=2,
      )
      with open(tf_l.name) as f:
        log_lines = f.readlines()
      # Headers plus fewer iterations than 50 // 2 + 1 = 26
      self.assertLess(len(log_lines), 26)

  def test_fit_logitboost(self) -> None:
    with tempfile.NamedTemporaryFile(delete=False) as tf:
      weights_file_path = tf.name
      self.addCleanup(os.remove, tf.name)
    with tempfile.NamedTemporaryFile(delete=False) as tf:
      log_file_path = tf.name
      self.addCleanup(os.remove, tf.name)
    X = jnp.array([[0, 1, 1, 1], [1, 1, 0, 1], [0, 0, 1, 1], [1, 0, 0, 1]])
    Y = jnp.array([-1, -1, 1, 1])
    rows, cols = jnp.where(X == 1)
    dataset = train.Dataset(rows, cols, Y)
    features = ['a', 'b', 'c']
    iters = 5
    out_span = 2
    scores = train.fit(
      dataset,
      dataset,
      features,
      iters,
      weights_file_path,
      log_file_path,
      out_span,
      algorithm='logitboost',
    )
    with open(weights_file_path) as f:
      weights = [line.split('\t') for line in f.read().splitlines() if line.strip()]
    top_feature = weights[0][0]
    self.assertEqual(
      top_feature,
      'b',
      msg='The most effective feature should be selected in LogitBoost.',
    )
    self.assertEqual(
      len(weights),
      iters,
      msg='The number of lines should equal to the iteration count.',
    )
    self.assertEqual(scores.shape[0], len(features))

  def test_fit_logitboost_weighted(self) -> None:
    with tempfile.NamedTemporaryFile(delete=False) as tf:
      weights_file_path = tf.name
      self.addCleanup(os.remove, tf.name)
    with tempfile.NamedTemporaryFile(delete=False) as tf:
      log_file_path = tf.name
      self.addCleanup(os.remove, tf.name)
    dataset = train.Dataset(jnp.array([0, 1]), jnp.array([0, 1]), jnp.array([10, 1]))
    features = ['a', 'b']
    train.fit(
      dataset,
      None,
      features,
      iters=1,
      weights_filename=weights_file_path,
      log_filename=log_file_path,
      out_span=1,
      algorithm='logitboost',
    )
    with open(weights_file_path) as f:
      weights = [line.split('\t') for line in f.read().splitlines() if line.strip()]
    self.assertEqual(weights[0][0], 'a')

  def test_fit_logitboost_learning_rate(self) -> None:
    with tempfile.NamedTemporaryFile(delete=False) as tf:
      weights_file_path = tf.name
      self.addCleanup(os.remove, tf.name)
    with tempfile.NamedTemporaryFile(delete=False) as tf:
      log_file_path = tf.name
      self.addCleanup(os.remove, tf.name)
    dataset = train.Dataset(jnp.array([0, 1]), jnp.array([0, 1]), jnp.array([1, 1]))
    features = ['a', 'b']
    scores = train.fit(
      dataset,
      None,
      features,
      iters=2,
      weights_filename=weights_file_path,
      log_filename=log_file_path,
      out_span=1,
      algorithm='logitboost',
      learning_rate=0.5,
    )
    self.assertEqual(scores.shape[0], len(features))

  def test_fit_invalid_learning_rate(self) -> None:
    with tempfile.NamedTemporaryFile(delete=False) as tf:
      weights_file_path = tf.name
      self.addCleanup(os.remove, tf.name)
    with tempfile.NamedTemporaryFile(delete=False) as tf:
      log_file_path = tf.name
      self.addCleanup(os.remove, tf.name)
    dataset = train.Dataset(jnp.array([0]), jnp.array([0]), jnp.array([1]))
    with self.assertRaises(ValueError) as cm:
      train.fit(
        dataset,
        None,
        ['a'],
        1,
        weights_file_path,
        log_file_path,
        1,
        algorithm='logitboost',
        learning_rate=0.0,
      )
    self.assertIn('learning_rate must be positive', str(cm.exception))

  def test_fit_invalid_algorithm(self) -> None:
    with tempfile.NamedTemporaryFile(delete=False) as tf:
      weights_file_path = tf.name
      self.addCleanup(os.remove, tf.name)
    with tempfile.NamedTemporaryFile(delete=False) as tf:
      log_file_path = tf.name
      self.addCleanup(os.remove, tf.name)
    dataset = train.Dataset(jnp.array([0]), jnp.array([0]), jnp.array([1]))
    with self.assertRaises(ValueError) as cm:
      train.fit(
        dataset,
        None,
        ['a'],
        1,
        weights_file_path,
        log_file_path,
        1,
        algorithm='unknown',
      )
    self.assertIn('Unknown algorithm', str(cm.exception))

  def test_fit_log_header_without_val(self) -> None:
    with tempfile.NamedTemporaryFile(delete=False) as tf:
      weights_file_path = tf.name
      self.addCleanup(os.remove, tf.name)
    with tempfile.NamedTemporaryFile(delete=False) as tf:
      log_file_path = tf.name
      self.addCleanup(os.remove, tf.name)
    dataset = train.Dataset(jnp.array([0, 1]), jnp.array([0, 1]), jnp.array([1, -1]))
    train.fit(dataset, None, ['a', 'b'], 1, weights_file_path, log_file_path, 1)
    with open(log_file_path) as f:
      log = [line.split('\t') for line in f.read().splitlines() if line.strip()]
    self.assertEqual(
      log[0],
      ['iter', 'train_accuracy', 'train_precision', 'train_recall', 'train_fscore'],
    )
    self.assertEqual(len(log[1]), 5)

  def test_fit_invalid_iters(self) -> None:
    with tempfile.NamedTemporaryFile(delete=False) as tf:
      weights_file_path = tf.name
      self.addCleanup(os.remove, tf.name)
    with tempfile.NamedTemporaryFile(delete=False) as tf:
      log_file_path = tf.name
      self.addCleanup(os.remove, tf.name)
    dataset = train.Dataset(jnp.array([0]), jnp.array([0]), jnp.array([1]))
    with self.assertRaises(ValueError) as cm:
      train.fit(dataset, None, ['a'], 0, weights_file_path, log_file_path, 1)
    self.assertIn('iters must be positive', str(cm.exception))

  def test_fit_invalid_out_span(self) -> None:
    with tempfile.NamedTemporaryFile(delete=False) as tf:
      weights_file_path = tf.name
      self.addCleanup(os.remove, tf.name)
    with tempfile.NamedTemporaryFile(delete=False) as tf:
      log_file_path = tf.name
      self.addCleanup(os.remove, tf.name)
    dataset = train.Dataset(jnp.array([0]), jnp.array([0]), jnp.array([1]))
    with self.assertRaises(ValueError) as cm:
      train.fit(dataset, None, ['a'], 1, weights_file_path, log_file_path, 0)
    self.assertIn('out_span must be positive', str(cm.exception))


class TestExtractFeatures(unittest.TestCase):
  def test_with_standard_setup(self) -> None:
    with tempfile.NamedTemporaryFile(delete=False) as tf:
      entries_file_path = tf.name
      self.addCleanup(os.remove, tf.name)
    with open(entries_file_path, 'w') as f:
      f.write('1\tfoo\tbar\n-1\tfoo\n1\tfoo\tbar\tbaz\n1\tbar\tfoo\n-1\tbaz\tqux\n')
    result = train.extract_features(entries_file_path, 1)
    self.assertEqual(result, ['foo', 'bar', 'baz'])

  def test_with_redundant_breaks(self) -> None:
    with tempfile.NamedTemporaryFile(delete=False) as tf:
      entries_file_path = tf.name
      self.addCleanup(os.remove, tf.name)
    with open(entries_file_path, 'w') as f:
      f.write('1\tfoo\tbar\n-1\tfoo\n1\tfoo\tbar\tbaz\n\n1\tbar\tfoo\n\n-1\tbaz\tqux\n')
    result = train.extract_features(entries_file_path, 1)
    self.assertEqual(result, ['foo', 'bar', 'baz'])

  def test_with_weighted_entries(self) -> None:
    with tempfile.NamedTemporaryFile(delete=False) as tf:
      entries_file_path = tf.name
      self.addCleanup(os.remove, tf.name)
    with open(entries_file_path, 'w') as f:
      f.write('2\tfoo\n-3\tbar\n1\tbaz\n')
    result = train.extract_features(entries_file_path, 1)
    self.assertEqual(result, ['bar', 'foo'])

  def test_utf8_feature_extraction(self) -> None:
    with tempfile.NamedTemporaryFile(delete=False) as tf:
      train_data_path = tf.name
      self.addCleanup(os.remove, tf.name)
    with open(train_data_path, 'w', encoding='utf-8') as f:
      f.write('1\tBW1:今日\tBC1:は\n-1\tBW1:今日\n1\tBW1:今日\tBC1:は\tBC2:良\n')
    features = train.extract_features(train_data_path, 1)
    self.assertEqual(features, ['BW1:今日', 'BC1:は'])


class TestLoadDataset(unittest.TestCase):
  def test_with_standard_setup(self) -> None:
    with tempfile.NamedTemporaryFile(delete=False) as tf:
      entries_file_path = tf.name
      self.addCleanup(os.remove, tf.name)
    with open(entries_file_path, 'w') as f:
      f.write('1\tfoo\tbar\n-1\tfoo\n1\tfoo\tbar\tbaz\n1\tbar\tfoo\n-1\tbaz\tqux\n')
    result = train.load_dataset(entries_file_path, {'foo': 0, 'bar': 1, 'baz': 2})
    self.assertEqual(result.X_rows.tolist(), [0, 0, 1, 2, 2, 2, 3, 3, 4])
    self.assertEqual(result.X_cols.tolist(), [0, 1, 0, 0, 1, 2, 1, 0, 2])
    self.assertEqual(result.Y.tolist(), [1, -1, 1, 1, -1])

  def test_with_redundant_breaks(self) -> None:
    with tempfile.NamedTemporaryFile(delete=False) as tf:
      entries_file_path = tf.name
      self.addCleanup(os.remove, tf.name)
    with open(entries_file_path, 'w') as f:
      f.write('1\tfoo\tbar\n-1\tfoo\n1\tfoo\tbar\tbaz\n\n1\tbar\tfoo\n\n-1\tbaz\tqux\n')
    result = train.load_dataset(entries_file_path, {'foo': 0, 'bar': 1, 'baz': 2})
    self.assertEqual(result.X_rows.tolist(), [0, 0, 1, 2, 2, 2, 3, 3, 4])
    self.assertEqual(result.X_cols.tolist(), [0, 1, 0, 0, 1, 2, 1, 0, 2])
    self.assertEqual(result.Y.tolist(), [1, -1, 1, 1, -1])

  def test_with_weighted_samples(self) -> None:
    with tempfile.NamedTemporaryFile(delete=False) as tf:
      entries_file_path = tf.name
      self.addCleanup(os.remove, tf.name)
    with open(entries_file_path, 'w') as f:
      f.write('1\tfoo\tbar\n-14\tfoo\n10\tfoo\tbar\tbaz\n-11\tbar\tfoo\n-1\tbaz\tqux\n')
    result = train.load_dataset(entries_file_path, {'foo': 0, 'bar': 1, 'baz': 2})
    self.assertEqual(result.X_rows.tolist(), [0, 0, 1, 2, 2, 2, 3, 3, 4])
    self.assertEqual(result.X_cols.tolist(), [0, 1, 0, 0, 1, 2, 1, 0, 2])
    self.assertEqual(result.Y.tolist(), [1, -14, 10, -11, -1])
