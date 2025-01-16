# Copyright 2024 Google LLC
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
"""All text signals including OSS and internal ones."""

# pylint: disable=g-importing-member
# pylint: disable=unused-import
from ml_metrics._src.signals.text import alphabetical_char_count
from ml_metrics._src.signals.text import average_word_length
from ml_metrics._src.signals.text import exact_match
from ml_metrics._src.signals.text import is_all_whitespace
from ml_metrics._src.signals.text import non_ascii_char_count
from ml_metrics._src.signals.text import reference_in_sample_match
from ml_metrics._src.signals.text import reference_startswith_sample_match
from ml_metrics._src.signals.text import sample_in_reference_match
from ml_metrics._src.signals.text import sample_startswith_reference_match
from ml_metrics._src.signals.text import token_count
from ml_metrics._src.signals.text import token_match_rate
from ml_metrics._src.signals.text import word_count
