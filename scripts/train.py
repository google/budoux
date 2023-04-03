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
"""Runs model training and exports the learned scores to build a model."""

import argparse
import array
import typing
from collections import Counter
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

EPS = np.finfo(float).eps  # type: np.floating[typing.Any]
DEFAULT_OUTPUT_NAME = 'weights.txt'
DEFAULT_LOG_NAME = 'train.log'
DEFAULT_FEATURE_THRES = 10
DEFAULT_ITERATION = 10000
DEFAULT_OUT_SPAN = 100
ArgList = typing.Optional[typing.List[str]]


class Result(NamedTuple):
  tp: int
  tn: int
  fp: int
  fn: int
  accuracy: float
  precision: float
  recall: float
  fscore: float


def preprocess(
    entries_filename: str, feature_thres: int
) -> typing.Tuple[typing.Any, typing.Any, typing.Any, typing.List[str]]:
  """Loads entries and translates them into JAX arrays. The boolean matrix of
  the input data is represented by row indices and column indices of True values
  instead of the matrix itself for memory efficiency, assuming the matrix is
  highly sparse. Row and column indices are not guaranteed to be sorted.

  Args:
    entries_filename (str): A file path to the entries file.
    feature_thres (str): A threshold to filter out features whose frequency is
      below the given value.

  Returns:
    A tuple of following items:
    - rows (JAX array): Row indices of True values in the input data.
    - cols (JAX array): Column indices of True values in the input data.
    - Y (JAX array): The target output data.
    - features (List[str]): The list of features.
  """
  features_counter: typing.Counter[str] = Counter()
  X = []
  Y = array.array('B')
  with open(entries_filename) as f:
    for row in f:
      cols = row.strip().split('\t')
      if len(cols) < 2:
        continue
      Y.append(cols[0] == '1')
      X.append(cols[1:])
      features_counter.update(cols[1:])
  features = [
      item[0]
      for item in features_counter.most_common()
      if item[1] > feature_thres
  ]
  feature_index = dict([(feature, i) for i, feature in enumerate(features)])
  rows = array.array('I')
  cols = array.array('I')  # type: ignore
  for i, x in enumerate(X):
    hit_indices = [feature_index[feat] for feat in x if feat in feature_index]
    rows.extend(i for _ in range(len(hit_indices)))
    cols.extend(hit_indices)  # type: ignore
  return jnp.asarray(rows), jnp.asarray(cols), jnp.asarray(
      Y, dtype=bool), features


def split_data(
    rows: npt.NDArray[np.int32],
    cols: npt.NDArray[np.int32],
    Y: npt.NDArray[np.bool_],
    split_ratio: float = .9
) -> typing.Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32],
                  npt.NDArray[np.int32], npt.NDArray[np.int32],
                  npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
  """Splits a dataset into a training dataset and a test dataset.

  Args:
    rows (numpy.ndarray): Row indices of True values in the input data.
    cols (numpy.ndarray): Column indices of True values in the input data.
    Y (numpy.ndarray): The target output.
    split_ratio (float, optional): The split ratio for the training dataset.
      The value should be between 0 and 1. The default value is 0.9 (=90% for
      training).

  Returns:
    A tuple of:
    - rows_train (numpy.ndarray): Row indices of True values in the training input data.
    - cols_train (numpy.ndarray): Column indices of True values in the training input data.
    - rows_test (numpy.ndarray): Row indices of True values in the test input data.
    - cols_test (numpy.ndarray): Column indices of True values in the test input data.
    - Y_train (numpy.ndarray): The training target output.
    - Y_test (numpy.ndarray): The test target output.
  """
  thres = int(Y.shape[0] * split_ratio)
  return (rows[rows < thres], cols[rows < thres], rows[rows >= thres] - thres,
          cols[rows >= thres], Y[:thres], Y[thres:])


@partial(jax.jit, static_argnums=[3])
def pred(scores: npt.NDArray[np.float32], rows: npt.NDArray[np.int32],
         cols: npt.NDArray[np.int32], N: int) -> npt.NDArray[np.bool_]:
  """Predicts the target output from the learned scores and input entries.

  Args:
    scores (numpy.ndarray): Contribution scores of features.
    rows (numpy.ndarray): Row indices of True values in the input.
    cols (numpy.ndarray): Column indices of True values in the input.
    N (int): The number of input entries.

  Returns:
    res (numpy.ndarray): A prediction of the target.
  """
  # This is equivalent to scores.dot(2X - 1) = 2 * scores.dot(X) - scores.sum()
  # but in a sparse matrix-friendly way.
  r: npt.NDArray[np.float32] = 2 * jax.ops.segment_sum(
      scores.take(cols), rows, N) - scores.sum()
  return r > 0


@jax.jit
def get_metrics(pred: npt.NDArray[np.bool_],
                actual: npt.NDArray[np.bool_]) -> Result:
  """Gets evaluation metrics from the prediction and the actual target.

  Args:
    pred (numpy.ndarray): A prediction of the target.
    actual (numpy.ndarray): The actual target.

  Returns:
    result (Result): A result.
  """
  tp: int = jnp.sum(jnp.logical_and(pred == 1, actual == 1))  # type: ignore
  tn: int = jnp.sum(jnp.logical_and(pred == 0, actual == 0))  # type: ignore
  fp: int = jnp.sum(jnp.logical_and(pred == 1, actual == 0))  # type: ignore
  fn: int = jnp.sum(jnp.logical_and(pred == 0, actual == 1))  # type: ignore
  accuracy = (tp + tn) / (tp + tn + fp + fn)
  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  return Result(
      tp=tp,
      tn=tn,
      fp=fp,
      fn=fn,
      accuracy=accuracy,
      precision=precision,
      recall=recall,
      fscore=2 * precision * recall / (precision + recall),
  )


@jax.jit
def update(
    w: npt.NDArray[np.float32], scores: typing.Any, rows: npt.NDArray[np.int32],
    cols: npt.NDArray[np.int32], Y: npt.NDArray[np.bool_]
) -> typing.Tuple[typing.Any, typing.Any, int, float]:
  """Calculates the new weight vector and the contribution scores.

  Args:
    w (numpy.ndarray): A weight vector.
    scores (JAX array): Contribution scores of features.
    rows (numpy.ndarray): Row indices of True values in the input data.
    cols (numpy.ndarray): Column indices of True values in the input data.
    Y (numpy.ndarray): The target output.


  Returns:
    A tuple of following items:
    - w (numpy.ndarray): The new weight vector.
    - scores (JAX array): The new contribution scores.
    - best_feature_index (int): The index of the best feature.
    - score (float): The newly added score for the best feature.
  """
  N = w.shape[0]
  M = scores.shape[0]
  # This is quivalent to w.dot(Y[:, None] ^ X). Note that y ^ x = y + x - 2yx,
  # hence w.dot(y ^ x) = w.dot(y) - w(2y - 1).dot(x).
  # `segment_sum` is used to implement sparse matrix-friendly dot products.
  res = w.dot(Y) - jax.ops.segment_sum((w * (2 * Y - 1)).take(rows), cols, M)
  err = 0.5 - jnp.abs(res - 0.5)
  best_feature_index: int = err.argmin()
  positivity: bool = res.at[best_feature_index].get() < 0.5
  err_min = err.at[best_feature_index].get()
  amount: float = jnp.log((1 - err_min) / (err_min + EPS))  # type: ignore

  # This is equivalent to X_best = X[:, best_feature_index]
  X_best = jnp.zeros(
      N, dtype=bool).at[jnp.where(cols == best_feature_index, rows, N)].set(
          True, mode='drop')
  w = w * jnp.exp(amount * (Y ^ X_best == positivity))
  w = w / w.sum()
  score = amount * (2 * positivity - 1)
  scores = scores.at[best_feature_index].add(score)
  return w, scores, best_feature_index, score


def fit(rows_train: npt.NDArray[np.int32], cols_train: npt.NDArray[np.int32],
        rows_test: npt.NDArray[np.int32], cols_test: npt.NDArray[np.int32],
        Y_train: npt.NDArray[np.bool_], Y_test: npt.NDArray[np.bool_],
        features: typing.List[str], iters: int, weights_filename: str,
        log_filename: str, out_span: int) -> typing.Any:
  """Trains an AdaBoost binary classifier.

  Args:
    row_train (numpy.ndarray): Row indices of True values in the training input data.
    col_train (numpy.ndarray): Column indices of True values in the training input data.
    row_test (numpy.ndarray): Row indices of True values in the test input data.
    col_test (numpy.ndarray): Column indices of True values in the test input data.
    Y_train (numpy.ndarray): The training target output.
    Y_test (numpy.ndarray): The test target output.
    features (List[str]): Features, which correspond to the columns of entries.
    iters (int): A number of training iterations.
    weights_filename (str): A file path to write the learned weights.
    log_filename (str): A file path to log the accuracy along with training.
    out_span (int): Iteration span to output metics and weights.

  Returns:
    scores (Any): The contribution scores.
  """
  with open(weights_filename, 'w') as f:
    f.write('')
  with open(log_filename, 'w') as f:
    f.write(
        'iter\ttrain_accuracy\ttrain_precision\ttrain_recall\ttrain_fscore\t'
        'test_accuracy\ttest_precision\ttest_recall\ttest_fscore\n')
  print('Outputting learned weights to %s ...' % (weights_filename))

  M = len(features)
  scores = jnp.zeros(M)
  feature_score_buffer: typing.List[typing.Tuple[str, float]] = []
  N_train = Y_train.shape[0]
  N_test = Y_test.shape[0]
  w = jnp.ones(N_train) / N_train

  def output_progress(t: int) -> None:
    print('=== %s ===' % t)
    with open(weights_filename, 'a') as f:
      f.write('\n'.join('%s\t%.6f' % p for p in feature_score_buffer) + '\n')
    feature_score_buffer.clear()
    pred_train = pred(scores, rows_train, cols_train, N_train)
    pred_test = pred(scores, rows_test, cols_test, N_test)
    metrics_train = get_metrics(pred_train, Y_train)
    metrics_test = get_metrics(pred_test, Y_test)
    print()
    print('train accuracy:\t%.5f' % metrics_train.accuracy)
    print('train prec.:\t%.5f' % metrics_train.precision)
    print('train recall:\t%.5f' % metrics_train.recall)
    print('train fscore:\t%.5f' % metrics_train.fscore)
    print()
    print('test accuracy:\t%.5f' % metrics_test.accuracy)
    print('test prec.:\t%.5f' % metrics_test.precision)
    print('test recall:\t%.5f' % metrics_test.recall)
    print('test fscore:\t%.5f' % metrics_test.fscore)
    print()
    with open(log_filename, 'a') as f:
      f.write('%d\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n' % (
          t,
          metrics_train.accuracy,
          metrics_train.precision,
          metrics_train.recall,
          metrics_train.fscore,
          metrics_test.accuracy,
          metrics_test.precision,
          metrics_test.recall,
          metrics_test.fscore,
      ))

  for t in range(iters):
    w, scores, best_feature_index, score = update(w, scores, rows_train,
                                                  cols_train, Y_train)
    w.block_until_ready()
    feature = features[best_feature_index]
    feature_score_buffer.append((feature, score))
    if (t + 1) % out_span == 0:
      output_progress(t + 1)
  if len(feature_score_buffer) > 0:
    output_progress(t + 1)
  return scores


def parse_args(test: ArgList = None) -> argparse.Namespace:
  """Parses commandline arguments.

  Args:
    test (typing.Optional[typing.List[str]], optional): Commandline args for
      testing. Defaults to None.

  Returns:
    argparse.Namespace: Parsed data of args.
  """
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      'encoded_train_data', help='File path for the encoded training data.')
  parser.add_argument(
      '-o',
      '--output',
      help=f'Output file path for the learned weights. (default: {DEFAULT_OUTPUT_NAME})',
      type=str,
      default=DEFAULT_OUTPUT_NAME)
  parser.add_argument(
      '--log',
      help=f'Output file path for the training log. (default: {DEFAULT_LOG_NAME})',
      type=str,
      default=DEFAULT_LOG_NAME)
  parser.add_argument(
      '--feature-thres',
      help=f'Threshold value of the minimum feature frequency. (default: {DEFAULT_FEATURE_THRES})',
      type=int,
      default=DEFAULT_FEATURE_THRES)
  parser.add_argument(
      '--iter',
      help=f'Number of iterations for training. (default: {DEFAULT_ITERATION})',
      type=int,
      default=DEFAULT_ITERATION)
  parser.add_argument(
      '--out-span',
      help=f'Iteration span to output metrics and weights. (default: {DEFAULT_OUT_SPAN})',
      type=int,
      default=DEFAULT_OUT_SPAN)
  if test is None:
    return parser.parse_args()
  else:
    return parser.parse_args(test)


def main() -> None:
  args = parse_args()
  data_filename: str = args.encoded_train_data
  weights_filename: str = args.output
  log_filename: str = args.log
  feature_thres = int(args.feature_thres)
  iterations = int(args.iter)
  out_span = int(args.out_span)

  X_rows, X_cols, Y, features = preprocess(data_filename, feature_thres)
  X_rows_train, X_cols_train, X_rows_test, X_cols_test, Y_train, Y_test = split_data(
      X_rows, X_cols, Y)
  fit(X_rows_train, X_cols_train, X_rows_test, X_cols_test, Y_train, Y_test,
      features, iterations, weights_filename, log_filename, out_span)
  print('Training done. Export the model by passing %s to build_model.py' %
        (weights_filename))


if __name__ == '__main__':
  main()
