"""Tests for image."""

import io

from ml_metrics._src.signals import image
from PIL import Image

from absl.testing import absltest


class ImageTest(absltest.TestCase):

  def test_content_metadata(self):
    single_black_pixel_image = Image.new('RGB', (1, 1))
    with io.BytesIO() as output:
      single_black_pixel_image.save(output, format='PNG')
      single_black_pixel_image_bytes = output.getvalue()

    expected_metadata = {
        'mode': 'RGB',
        'format': 'PNG',
        'pixel_width': 1,
        'pixel_height': 1,
        'megapixel_resolution': 1 / image.MEGAPIXELS,
        'aspect_ratio': 1.0,
    }

    metadata = image.content_metadata(single_black_pixel_image_bytes)
    self.assertEqual(metadata, expected_metadata)


if __name__ == '__main__':
  absltest.main()
