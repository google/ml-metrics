# Copyright 2024 Google LLC
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

from absl.testing import absltest
from ml_metrics._src.chainables import transform
from ml_metrics._src.chainables import tree_fns
from ml_metrics._src.utils import df_utils
import pandas as pd


class DfUtilsTest(absltest.TestCase):

  def test_as_dataframe_default(self):
    key = transform.MetricKey(
        metrics='m2', slice=tree_fns.SliceKey(('f1',), ('a',))
    )
    agg_result = {'m1': 1, key: 2}
    df = df_utils.metrics_to_df(agg_result)
    pd.testing.assert_frame_equal(
        df,
        pd.DataFrame({
            'metric_name': ['m1', 'm2'],
            'slice': ['overall', 'f1=a'],
            'value': [1, 2],
        }),
    )


if __name__ == '__main__':
  absltest.main()
