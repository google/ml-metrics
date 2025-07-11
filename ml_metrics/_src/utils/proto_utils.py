# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Proto utils."""
from collections.abc import Iterable
from typing import Any
from absl import logging
from ml_metrics._src.tools.telemetry import telemetry
import more_itertools as mit
import numpy as np
from tensorflow.core.example import example_pb2

_ExampleOrBytes = bytes | example_pb2.Example


def _maybe_deserialize(ex: _ExampleOrBytes) -> example_pb2.Example:
  if isinstance(ex, bytes):
    return example_pb2.Example.FromString(ex)
  if isinstance(ex, example_pb2.Example):
    return ex
  raise TypeError('Unsupported type: %s' % type(ex))


@telemetry.function_monitor(api='ml_metrics', category=telemetry.CATEGORY.UTIL)
def tf_examples_to_dict(examples: Iterable[_ExampleOrBytes] | _ExampleOrBytes):
  """Parses a serialized tf.train.Example to a dict."""
  single_example = False
  if isinstance(examples, (bytes, example_pb2.Example)):
    single_example = True
    examples = [examples]
  examples = (_maybe_deserialize(ex) for ex in examples)
  examples = mit.peekable(examples)
  if (head := examples.peek(None)) is None:
    return {}

  result = {k: [] for k in head.features.feature}
  for ex in examples:
    missing = set(result)
    for key, feature in ex.features.feature.items():
      missing.remove(key)
      value = getattr(feature, feature.WhichOneof('kind')).value
      if value and isinstance(value[0], bytes):
        try:
          value = [v.decode() for v in value]
        except UnicodeDecodeError:
          logging.info(
              'chainable: %s',
              f'Failed to decode for {key}, forward the raw bytes.',
          )
      result[key].extend(value)
    if missing:
      raise ValueError(
          f'Missing keys: {missing}, expecting {set(result)}, got {ex=}'
      )
  result = {k: v for k, v in result.items()}
  # Scalar value in a single example will be returned with the scalar directly.
  if single_example and all(len(v) == 1 for v in result.values()):
    result = {k: v[0] for k, v in result.items()}
  return result


@telemetry.function_monitor(api='ml_metrics', category=telemetry.CATEGORY.UTIL)
def dict_to_tf_example(data: dict[str, Any]) -> example_pb2.Example:
  """Creates a tf.Example from a dictionary."""
  example = example_pb2.Example()
  for key, value in data.items():
    if isinstance(value, (str, bytes, np.floating, float, int, np.integer)):
      value = [value]
    feature = example.features.feature
    if isinstance(value[0], str):
      for v in value:
        assert isinstance(v, str), f'bad str type: {value}'
        feature[key].bytes_list.value.append(v.encode())
    elif isinstance(value[0], bytes):
      feature[key].bytes_list.value.extend(value)
    elif isinstance(value[0], (int, np.integer)):
      feature[key].int64_list.value.extend(value)
    elif isinstance(value[0], (float, np.floating)):
      feature[key].float_list.value.extend(value)
    else:
      raise TypeError(f'Value for "{key}" is not a supported type.')
  return example
