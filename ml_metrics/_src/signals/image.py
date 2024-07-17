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
"""Samplewise scoring metrics for image."""

import io

from PIL import Image


MEGAPIXELS = 1024 * 1024


def content_metadata(image_bytes: bytes) -> dict[str, int | float | str]:
  """Extracts the content metadata of an image."""

  img = Image.open(io.BytesIO(image_bytes))
  width, height = img.size
  return {
      'mode': img.mode,
      'format': img.format,
      'pixel_width': width,
      'pixel_height': height,
      'megapixel_resolution': width * height / MEGAPIXELS,
      'aspect_ratio': width / height if height > 0 else 0.0,
  }
