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
from google.protobuf import text_format
from ml_metrics._src.utils import proto_utils
from ml_metrics._src.utils import test_utils
import tensorflow as tf
from absl.testing import absltest
from absl.testing import parameterized
from tensorflow.core.example import example_pb2


EXAMPLE_1 = text_format.Parse(
    """
features {
  feature {
    key: "bytes"
    value {
      bytes_list {
        value: "ab"
      }
    }
  }
  feature {
    key: "bytes_arr"
    value {
      bytes_list {
        value: "cd"
        value: "ef"
      }
    }
  }
  feature {
    key: "int64"
    value {
      int64_list {
        value: 1
      }
    }
  }
  feature {
    key: "int64_arr"
    value {
      int64_list {
        value: 2
        value: 3
      }
    }
  }
  feature {
    key: "float"
    value {
      float_list {
        value: 1.5
      }
    }
  }
  feature {
    key: "float_arr"
    value {
      float_list {
        value: 2.5
        value: 3.5
      }
    }
  }
}
""",
    example_pb2.Example(),
)


EXAMPLE_2 = text_format.Parse(
    """
features {
  feature {
    key: "bytes"
    value {
      bytes_list {
        value: "mn"
      }
    }
  }
  feature {
    key: "bytes_arr"
    value {
      bytes_list {
        value: "op"
        value: "qr"
      }
    }
  }
  feature {
    key: "int64"
    value {
      int64_list {
        value: 4
      }
    }
  }
  feature {
    key: "int64_arr"
    value {
      int64_list {
        value: 5
        value: 6
      }
    }
  }
  feature {
    key: "float"
    value {
      float_list {
        value: 11.5
      }
    }
  }
  feature {
    key: "float_arr"
    value {
      float_list {
        value: 12.5
        value: 13.5
      }
    }
  }
}
""",
    example_pb2.Example(),
)


class TFExampleTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='tf_example',
          example=EXAMPLE_1,
      ),
      dict(
          testcase_name='serialized_tf_example',
          example=EXAMPLE_1.SerializeToString(),
      ),
  )
  def test_single_example_to_dict(self, example):
    actual = proto_utils.tf_examples_to_dict(example)
    expected = {
        'bytes': [b'ab'],
        'bytes_arr': [b'cd', b'ef'],
        'int64': [1],
        'int64_arr': [2, 3],
        'float': [1.5],
        'float_arr': [2.5, 3.5],
    }
    test_utils.assert_nested_container_equal(self, expected, actual, places=6)

  @parameterized.named_parameters(
      dict(
          testcase_name='batch_tf_examples',
          examples=[EXAMPLE_1, EXAMPLE_2],
      ),
      dict(
          testcase_name='batch_serialized_tf_examples',
          examples=[
              EXAMPLE_1.SerializeToString(),
              EXAMPLE_2.SerializeToString(),
          ],
      ),
      dict(
          testcase_name='batch_mixed_tf_examples',
          examples=[EXAMPLE_1.SerializeToString(), EXAMPLE_2],
      ),
  )
  def test_batched_examples_to_dict(self, examples):
    actual = proto_utils.tf_examples_to_dict(examples)
    expected = {
        'bytes': [[b'ab'], [b'mn']],
        'bytes_arr': [[b'cd', b'ef'], [b'op', b'qr']],
        'int64': [[1], [4]],
        'int64_arr': [[2, 3], [5, 6]],
        'float': [[1.5], [11.5]],
        'float_arr': [[2.5, 3.5], [12.5, 13.5]],
    }
    test_utils.assert_nested_container_equal(self, expected, actual, places=6)

  def test_batched_single_example_to_dict(self):
    actual = proto_utils.tf_examples_to_dict([EXAMPLE_1])
    expected = {
        'bytes': [[b'ab']],
        'bytes_arr': [[b'cd', b'ef']],
        'int64': [[1]],
        'int64_arr': [[2, 3]],
        'float': [[1.5]],
        'float_arr': [[2.5, 3.5]],
    }
    test_utils.assert_nested_container_equal(self, expected, actual, places=6)

  def test_missing_features_example_to_dict(self):
    example_missing_features = text_format.Parse(
        """
    features {
      feature {
        key: "bytes"
        value {
          bytes_list {
            value: "xy"
          }
        }
      }
    }
    """,
        example_pb2.Example(),
    )

    with self.assertRaisesRegex(
        ValueError, 'All examples must have the same features'
    ):
      _ = proto_utils.tf_examples_to_dict([example_missing_features, EXAMPLE_1])

  def test_empty_example_to_dict(self):
    self.assertEmpty(proto_utils.tf_examples_to_dict([]))

  def test_dict_to_tf_example(self):
    data = {
        'bytes_scalar': b'a',
        'str_scalar': 'b',
        'int64_scalar': 1,
        'flaot_scalar': 2.1,
        'bytes_list': [b'cd', b'ef'],
        'str_list': ['gh', 'ij'],
        'int64_list': [2, 3],
        'float_list': [1, 3.5],
    }
    expected = text_format.Parse(
        """
    features {
      feature {
        key: "bytes_scalar"
        value {
          bytes_list {
            value: "a"
          }
        }
      }
      feature {
        key: "str_scalar"
        value {
          bytes_list {
            value: "b"
          }
        }
      }
      feature {
        key: "int64_scalar"
        value {
          int64_list {
            value: 1
          }
        }
      }
      feature {
        key: "flaot_scalar"
        value {
          float_list {
            value: 2.1
          }
        }
      }
      feature {
        key: "bytes_list"
        value {
          bytes_list {
            value: "cd"
            value: "ef"
          }
        }
      }
      feature {
        key: "str_list"
        value {
          bytes_list {
            value: "gh"
            value: "ij"
          }
        }
      }
      feature {
        key: "int64_list"
        value {
          int64_list {
            value: 2
            value: 3
          }
        }
      }
      feature {
        key: "float_list"
        value {
          float_list {
            value: 1.0
            value: 3.5
          }
        }
      }
    }
    """,
        example_pb2.Example(),
    )
    actual = proto_utils.dict_to_tf_example(data)
    self.assertProtoEquals(expected, actual)

  def test_dict_to_tf_example_key_with_empty_list(self):
    data = {'int64_list': 1, 'empty_list': []}
    expected = text_format.Parse(
        """
    features {
      feature {
        key: "int64_list"
        value {
          int64_list {
            value: 1
          }
        }
      }
    }
    """,
        example_pb2.Example(),
    )
    actual = proto_utils.dict_to_tf_example(data)
    self.assertProtoEquals(expected, actual)

  def test_dict_to_tf_example_bad_str_type(self):
    data = {'str_arr': ['abc', b'def']}
    with self.assertRaisesRegex(AssertionError, 'bad str type'):
      _ = proto_utils.dict_to_tf_example(data)

  @parameterized.named_parameters(
      dict(
          testcase_name='bytes',
          data={'bad_type': [b'ab', 'cd']},
      ),
      dict(
          testcase_name='float',
          data={'bad_type': [1.0, 'a']},
      ),
      dict(
          testcase_name='int64',
          data={'bad_type': [1, 'b']},
      ),
  )
  def test_dict_to_tf_example_inconsistent_types(self, data):
    # This test is required as the logic to determine the type of the feature
    # list is based on the first value of the list.
    with self.assertRaises(Exception):
      _ = proto_utils.dict_to_tf_example(data)

  def test_dict_to_tf_example_unsupported_type(self):
    data = {'bad_type': [example_pb2.Example()]}
    with self.assertRaisesRegex(
        TypeError, 'Values for "bad_type" is not a supported type.'
    ):
      _ = proto_utils.dict_to_tf_example(data)


if __name__ == '__main__':
  absltest.main()
