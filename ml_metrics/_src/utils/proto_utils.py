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
from typing import Any
from absl import logging
import deprecated
from ml_metrics._src.tools.telemetry import telemetry
import numpy as np
from tensorflow.core.example import example_pb2

_ExampleOrBytes = bytes | example_pb2.Example


def _maybe_deserialize(ex: _ExampleOrBytes) -> example_pb2.Example:
  if isinstance(ex, bytes):
    return example_pb2.Example.FromString(ex)
  if isinstance(ex, example_pb2.Example):
    return ex
  raise TypeError('Unsupported type: %s' % type(ex))


@deprecated.deprecated('Use tf_example_to_dict instead.')
def tf_examples_to_dict(examples: _ExampleOrBytes):
  if isinstance(examples, example_pb2.Example):
    return tf_example_to_dict(examples)

  raise TypeError(
      'Mutliple examples are not supported, got %s' % type(examples)
  )


@telemetry.function_monitor(api='ml_metrics', category=telemetry.CATEGORY.UTIL)
def tf_example_to_dict(
    example: _ExampleOrBytes,
    *,
    unwrap_scalar: bool = True,
    decode_bytes_as_str: bool = True,
):
  """Parses a serialized tf.train.Example to a dict."""
  example = _maybe_deserialize(example)

  result = {}
  for key, feature in example.features.feature.items():
    value = getattr(feature, feature.WhichOneof('kind')).value
    if value and isinstance(value[0], bytes) and decode_bytes_as_str:
      try:
        value = [v.decode() for v in value]
      except UnicodeDecodeError:
        logging.info(
            'chainable: %s',
            f'Failed to decode for {key}, forward the raw bytes.',
        )
    result[key] = value[0] if unwrap_scalar and len(value) == 1 else value
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
