import tempfile
import unittest
from pathlib import Path

from PIL import Image

from utils.utils import ImageFolderDataset


class ImageFolderDatasetTests(unittest.TestCase):
    def test_skips_invalid_images_and_loads_valid_ones(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            (tmp_path / 'bad.jpg').write_bytes(b'not-an-image')

            valid_path = tmp_path / 'good.png'
            Image.new('RGB', (8, 8), color='red').save(valid_path)

            dataset = ImageFolderDataset(str(tmp_path))

            self.assertEqual(len(dataset), 1)
            sample = dataset[0]
            self.assertEqual(sample.size[0], 8)
            self.assertEqual(sample.size[1], 8)


if __name__ == '__main__':
    unittest.main()
