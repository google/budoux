# Copyright 2023 Google LLC
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
"""Finetunes a BudouX model with the given training dataset.

Example usage:

$ python finetune.py train_data.txt base_model.json -o weights.txt --val_data=val_data.txt
"""

import argparse
import array
import json
import typing
from collections import OrderedDict

from jax import Array, grad, jit
from jax import numpy as jnp

EPSILON: float = jnp.finfo(float).eps
DEFAULT_OUTPUT_NAME = 'finetuned-weights.txt'
DEFAULT_NUM_ITERS = 1000
DEFAULT_LOG_SPAN = 100
DEFAULT_LEARNING_RATE = 0.01


class NormalizedModel(typing.NamedTuple):
  features: typing.List[str]
  weights: Array


class Dataset(typing.NamedTuple):
  X: Array
  Y: Array


class Metrics(typing.NamedTuple):
  tp: int
  tn: int
  fp: int
  fn: int
  accuracy: float
  precision: float
  recall: float
  fscore: float
  loss: float


def load_model(file_path: str) -> NormalizedModel:
  """Loads a model as a pair of a features list and a normalized weight vector.

  Args:
    file_path: A file path for the model JSON file.

  Returns:
    A normalized model, which is a pair of a list of feature identifiers and a
    normalized weight vector.
  """
  with open(file_path) as f:
    model = json.load(f)
  model_flat = OrderedDict()
  for category in model:
    for item in model[category]:
      model_flat['%s:%s' % (category, item)] = model[category][item]
  weights = jnp.array(list(model_flat.values()))
  weights = weights / weights.std()
  weights = weights - weights.mean()
  keys = list(model_flat.keys())
  return NormalizedModel(keys, weights)


def load_dataset(file_path: str, model: NormalizedModel) -> Dataset:
  """Loads a dataset from the given file path.

  Args:
    file_path: A file path for the encoded data file.
    model: A normalized model.

  Returns:
    A dataset of inputs (X) and outputs (Y).
  """
  xs = []
  ys = array.array('B')
  with open(file_path) as f:
    for row in f:
      cols = row.strip().split('\t')
      if len(cols) < 2:
        continue
      ys.append(cols[0] == '1')
      xs.append(tuple(k in set(cols[1:]) for k in model.features))
  X = jnp.array(xs) * 2 - 1
  Y = jnp.array(ys)
  return Dataset(X, Y)


def cross_entropy_loss(weights: Array, x: Array, y: Array) -> Array:
  """Calcurates a cross entropy loss with a prediction by a sigmoid function.

  Args:
    weights: A weight vector.
    x: An input array.
    y: A target output array.

  Returns:
    A cross entropy loss.
  """
  pred = 1 / (1 + jnp.exp(-x.dot(weights)))
  return -jnp.mean(y * jnp.log(pred) + (1 - y) * jnp.log(1 - pred))


def get_metrics(weights: Array, dataset: Dataset) -> Metrics:
  """Gets evaluation metrics from the learned weight vector and the dataset.

  Args:
    weights: A weight vector.
    dataset: A dataset.

  Returns:
    result (Metrics): The metrics over the given weights and the dataset.
  """
  pred = dataset.X.dot(weights) > 0
  actual = dataset.Y
  tp: int = jnp.sum(jnp.logical_and(pred == 1, actual == 1))  # type: ignore
  tn: int = jnp.sum(jnp.logical_and(pred == 0, actual == 0))  # type: ignore
  fp: int = jnp.sum(jnp.logical_and(pred == 1, actual == 0))  # type: ignore
  fn: int = jnp.sum(jnp.logical_and(pred == 0, actual == 1))  # type: ignore
  loss: float = cross_entropy_loss(weights, dataset.X,
                                   dataset.Y)  # type: ignore
  accuracy = (tp + tn) / (tp + tn + fp + fn)
  precision = tp / (tp + fp + EPSILON)
  recall = tp / (tp + fn + EPSILON)
  fscore = 2 * precision * recall / (precision + recall + EPSILON)
  return Metrics(
      tp=tp,
      tn=tn,
      fp=fp,
      fn=fn,
      accuracy=accuracy,
      precision=precision,
      recall=recall,
      fscore=fscore,
      loss=loss,
  )


def fit(weights: Array,
        train_dataset: Dataset,
        iters: int,
        learning_rate: float,
        log_span: int,
        val_dataset: typing.Optional[Dataset] = None) -> Array:
  """Updates the weights with the given dataset.

  Args:
    weights: A weight vector.
    train_dataset: A train dataset.
    iters: A number of iterations.
    learning_rate: A learning rate.
    log_span: A span to log metrics.
    val_dataset: A validation dataset (optional).

  Returns:
    An updated weight vector.
  """
  grad_loss = jit(grad(cross_entropy_loss, argnums=0))
  for t in range(iters):
    weights = weights - learning_rate * grad_loss(weights, train_dataset.X,
                                                  train_dataset.Y)
    if (t + 1) % log_span != 0:
      continue
    metrics_train = jit(get_metrics)(weights, train_dataset)
    print()
    print('iter:\t%d' % (t + 1))
    print()
    print('train accuracy:\t%.5f' % metrics_train.accuracy)
    print('train prec.:\t%.5f' % metrics_train.precision)
    print('train recall:\t%.5f' % metrics_train.recall)
    print('train fscore:\t%.5f' % metrics_train.fscore)
    print('train loss:\t%.5f' % metrics_train.loss)
    print()

    if val_dataset is None:
      continue
    metrics_val = jit(get_metrics)(weights, val_dataset)
    print('val accuracy:\t%.5f' % metrics_val.accuracy)
    print('val prec.:\t%.5f' % metrics_val.precision)
    print('val recall:\t%.5f' % metrics_val.recall)
    print('val fscore:\t%.5f' % metrics_val.fscore)
    print('val loss:\t%.5f' % metrics_val.loss)
    print()
  return weights


def write_weights(file_path: str, weights: Array,
                  features: typing.List[str]) -> None:
  """Writes learned weights and corresponsing features to a file.

  Args:
    file_path: A file path for the weights file.
    weights: A weight vector.
    features: A list of feature identifiers.
  """
  with open(file_path, 'w') as f:
    f.write('\n'.join([
        '%s\t%.6f' % (feature, weights[i]) for i, feature in enumerate(features)
    ]))


def parse_args(
    test: typing.Optional[typing.List[str]] = None) -> argparse.Namespace:
  """Parses commandline arguments.

  Args:
    test (typing.Optional[typing.List[str]], optional): Commandline args for
      testing. Defaults to None.

  Returns:
    Parsed arguments (argparse.Namespace).
  """
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument(
      'train_data', help='File path for the encoded training data.')
  parser.add_argument('base_model', help='File path for the base model file.')
  parser.add_argument(
      '-o',
      '--output',
      help=f'File path for the output weights. (default: {DEFAULT_OUTPUT_NAME})',
      type=str,
      default=DEFAULT_OUTPUT_NAME)
  parser.add_argument(
      '--val-data', help='File path for the encoded validation data.', type=str)
  parser.add_argument(
      '--iters',
      help=f'Number of iterations for training. (default: {DEFAULT_NUM_ITERS})',
      type=int,
      default=DEFAULT_NUM_ITERS)
  parser.add_argument(
      '--log-span',
      help=f'Iteration span to print metrics. (default: {DEFAULT_LOG_SPAN})',
      type=int,
      default=DEFAULT_LOG_SPAN)
  parser.add_argument(
      '--learning-rate',
      help=f'Learning rate. (default: {DEFAULT_LEARNING_RATE})',
      type=float,
      default=DEFAULT_LEARNING_RATE)
  if test is None:
    return parser.parse_args()
  else:
    return parser.parse_args(test)


def main() -> None:
  args = parse_args()
  train_data_path: str = args.train_data
  base_model_path: str = args.base_model
  weights_path: str = args.output
  iters: int = args.iters
  log_span: int = args.log_span
  learning_rate: float = args.learning_rate
  val_data_path: typing.Optional[str] = args.val_data

  model = load_model(base_model_path)
  train_dataset = load_dataset(train_data_path, model)
  val_dataset = load_dataset(val_data_path, model) if val_data_path else None
  weights = fit(
      model.weights,
      train_dataset,
      iters=iters,
      log_span=log_span,
      learning_rate=learning_rate,
      val_dataset=val_dataset)
  write_weights(weights_path, weights, model.features)


if __name__ == '__main__':
  main()
