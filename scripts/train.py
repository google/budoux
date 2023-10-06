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

EPS: float = jnp.finfo(float).eps
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


class Dataset(NamedTuple):
  """A dataset tuple.

  X_rows (jax.Array): Row indices of True values in the input data.
  X_cols (jax.Array): Column indices of True values in the input data.
  Y (jax.Array): The target output.
  """
  X_rows: jax.Array
  X_cols: jax.Array
  Y: jax.Array


def extract_features(data_path: str, thres: int) -> typing.List[str]:
  """Extracts a features list from the given encoded data file. This filters out
     features whose number of occurrences does not exceed the threshold.

  Args:
    data_path (str): The path to the encoded data file that contains the
      features to be extracted, which is typically a training data file.
    thres (int): A threshold to filter out features  whose number of occurrences
      does not exceed the threshold.

  Returns:
    A list of features
  """
  counter: typing.Counter[str] = Counter()
  with open(data_path) as f:
    for row in f:
      cols = row.strip().split('\t')
      if len(cols) < 2:
        continue
      counter.update(cols[1:])
  return [item[0] for item in counter.most_common() if item[1] > thres]


def load_dataset(data_path: str, findex: typing.Dict[str, int]) -> Dataset:
  """Loads a dataset from the given encoded data file.

  Args:
    data_path (str): A file path for the encoded data file.
    findex (Dict[str, int]): A dictionary that maps a feature to its index.

  Returns:
    A dataset
  """
  Y = array.array('B')
  X_rows = array.array('I')
  X_cols = array.array('I')
  with open(data_path) as f:
    i = 0
    for row in f:
      cols = row.strip().split('\t')
      if len(cols) < 2:
        continue
      Y.append(cols[0] == '1')
      hit_indices = [findex[feat] for feat in cols[1:] if feat in findex]
      X_rows.extend(i for _ in range(len(hit_indices)))
      X_cols.extend(hit_indices)
      i += 1
  return Dataset(
      jnp.asarray(X_rows), jnp.asarray(X_cols), jnp.asarray(Y, dtype=bool))


def preprocess(
    train_data_path: str,
    feature_thres: int,
    val_data_path: typing.Optional[str] = None,
) -> typing.Tuple[Dataset, typing.List[str], typing.Optional[Dataset]]:
  """Loads entries and translates them into JAX arrays. The boolean matrix of
  the input data is represented by row indices and column indices of True values
  instead of the matrix itself for memory efficiency, assuming the matrix is
  highly sparse. Row and column indices are not guaranteed to be sorted.

  Args:
    train_data_path (str): A file path to the training data file.
    feature_thres (str): A threshold to filter out features whose number of
      occurances does not exceed the value.
    val_data_path (str, optional): A file path to the validation data file.

  Returns:
    A tuple of following items:
    - train_dataset (Dataset): The training dataset.
    - features (List[str]): The list of features.
    - val_dataset (Optional[Dataset]): The validation dataset.
        This becomes None if val_data_path is None.
  """
  features = extract_features(train_data_path, feature_thres)
  feature_index = dict((feature, i) for i, feature in enumerate(features))
  train_dataset = load_dataset(train_data_path, feature_index)
  val_dataset = load_dataset(val_data_path,
                             feature_index) if val_data_path else None
  return train_dataset, features, val_dataset


@partial(jax.jit, static_argnums=[3])
def pred(scores: jax.Array, rows: jax.Array, cols: jax.Array,
         N: int) -> jax.Array:
  """Predicts the target output from the learned scores and input entries.

  Args:
    scores (jax.Array): Contribution scores of features.
    rows (jax.Array): Row indices of True values in the input.
    cols (jax.Array): Column indices of True values in the input.
    N (int): The number of input entries.

  Returns:
    res (jax.Array): A prediction of the target.
  """
  # This is equivalent to scores.dot(2X - 1) = 2 * scores.dot(X) - scores.sum()
  # but in a sparse matrix-friendly way.
  r: jax.Array = 2 * jax.ops.segment_sum(scores.take(cols), rows,
                                         N) - scores.sum()
  return r > 0


@jax.jit
def get_metrics(pred: jax.Array, actual: jax.Array) -> Result:
  """Gets evaluation metrics from the prediction and the actual target.

  Args:
    pred (jax.Array): A prediction of the target.
    actual (jax.Array): The actual target.

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
def update(w: jax.Array, scores: jax.Array, rows: jax.Array, cols: jax.Array,
           Y: jax.Array) -> typing.Tuple[jax.Array, jax.Array, int, float]:
  """Calculates the new weight vector and the contribution scores.

  Args:
    w (jax.Array): A weight vector.
    scores (JAX array): Contribution scores of features.
    rows (jax.Array): Row indices of True values in the input data.
    cols (jax.Array): Column indices of True values in the input data.
    Y (jax.Array): The target output.


  Returns:
    A tuple of following items:
    - w (jax.Array): The new weight vector.
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
  best_feature_index: int = err.argmin()  # type: ignore
  positivity: bool = res.at[best_feature_index].get() < 0.5  # type: ignore
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


def fit(dataset_train: Dataset, dataset_val: typing.Optional[Dataset],
        features: typing.List[str], iters: int, weights_filename: str,
        log_filename: str, out_span: int) -> jax.Array:
  """Trains an AdaBoost binary classifier.

  Args:
    dataset_train (Dataset): A training dataset.
    dataset_val (Optional[Dataset]): A validation dataset.
    features (List[str]): Features, which correspond to the columns of entries.
    iters (int): A number of training iterations.
    weights_filename (str): A file path to write the learned weights.
    log_filename (str): A file path to log the accuracy along with training.
    out_span (int): Iteration span to output metics and weights.

  Returns:
    scores (jax.Array): The contribution scores.
  """
  with open(weights_filename, 'w') as f:
    f.write('')
  with open(log_filename, 'w') as f:
    f.write('iter\ttrain_accuracy\ttrain_precision\ttrain_recall\ttrain_fscore')
    if dataset_val:
      f.write('\ttest_accuracy\ttest_precision\ttest_recall\ttest_fscore')
    f.write('\n')
  print('Outputting learned weights to %s ...' % (weights_filename))

  M = len(features)
  scores = jnp.zeros(M)
  feature_score_buffer: typing.List[typing.Tuple[str, float]] = []
  N_train = dataset_train.Y.shape[0]
  N_test = dataset_val.Y.shape[0] if dataset_val else 0
  w = jnp.ones(N_train) / N_train

  def output_progress(t: int) -> None:
    with open(weights_filename, 'a') as f:
      f.write('\n'.join('%s\t%.6f' % p for p in feature_score_buffer) + '\n')
    feature_score_buffer.clear()

    print('=== %s ===' % t)
    print()

    with open(log_filename, 'a') as f:
      pred_train = pred(scores, dataset_train.X_rows, dataset_train.X_cols,
                        N_train)
      metrics_train = get_metrics(pred_train, dataset_train.Y)
      print('train accuracy:\t%.5f' % metrics_train.accuracy)
      print('train prec.:\t%.5f' % metrics_train.precision)
      print('train recall:\t%.5f' % metrics_train.recall)
      print('train fscore:\t%.5f' % metrics_train.fscore)
      print()
      f.write('%d\t%.5f\t%.5f\t%.5f\t%.5f' % (
          t,
          metrics_train.accuracy,
          metrics_train.precision,
          metrics_train.recall,
          metrics_train.fscore,
      ))

      if dataset_val:
        pred_test = pred(scores, dataset_val.X_rows, dataset_val.X_cols, N_test)
        metrics_test = get_metrics(pred_test, dataset_val.Y)
        print('test accuracy:\t%.5f' % metrics_test.accuracy)
        print('test prec.:\t%.5f' % metrics_test.precision)
        print('test recall:\t%.5f' % metrics_test.recall)
        print('test fscore:\t%.5f' % metrics_test.fscore)
        print()

        f.write('\t%.5f\t%.5f\t%.5f\t%.5f' % (
            metrics_test.accuracy,
            metrics_test.precision,
            metrics_test.recall,
            metrics_test.fscore,
        ))

      f.write('\n')

  for t in range(iters):
    w, scores, best_feature_index, score = update(w, scores,
                                                  dataset_train.X_rows,
                                                  dataset_train.X_cols,
                                                  dataset_train.Y)
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
  parser.add_argument(
      '--val-data', help='File path for the encoded validation data.', type=str)
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
  val_data: typing.Optional[str] = args.val_data

  dataset_train, features, dataset_val = preprocess(data_filename,
                                                    feature_thres, val_data)
  fit(dataset_train, dataset_val, features, iterations, weights_filename,
      log_filename, out_span)
  print('Training done. Export the model by passing %s to build_model.py' %
        (weights_filename))


if __name__ == '__main__':
  main()
