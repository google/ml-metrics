# Copyright 2025 Google LLC
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
"""Text metrics."""

# pylint: disable=g-importing-member
# pylint: disable=unused-import
# Eventually move things under /aggregates/text.py to metrics/text.py. Making
# the classes callable and replace the functions in metrics/text.py
from ml_metrics._src.aggregates.text import PatternFrequency
from ml_metrics._src.aggregates.text import TopKWordNGrams
from ml_metrics._src.metrics.text import avg_alphabetical_char_count
from ml_metrics._src.metrics.text import pattern_frequency
from ml_metrics._src.metrics.text import topk_word_ngrams
