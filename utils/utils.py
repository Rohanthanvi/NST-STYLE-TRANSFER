from torch.utils.data import Dataset
import os
from PIL import Image, ImageFile, UnidentifiedImageError
from torchvision import transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        super(ImageFolderDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.files = []

        for current_root, _, filenames in os.walk(root):
            for name in sorted(filenames):
                if not name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue

                self.files.append(os.path.join(current_root, name))

        if not self.files:
            raise ValueError(f'No valid images found in {root}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if not self.files:
            raise IndexError('Dataset is empty')

        for offset in range(len(self.files)):
            image_path = self.files[(idx + offset) % len(self.files)]

            try:
                with Image.open(image_path) as image:
                    image = image.convert('RGB')

                    if self.transform:
                        image = self.transform(image)

                    return image
            except (IOError, OSError, UnidentifiedImageError, ValueError):
                continue

        raise RuntimeError(f'Failed to load any image from {self.root}')
    
def get_transform(size, crop, final_size):
    transform_list = []

    if size > 0:
        transform_list.append(transforms.Resize(max(size, final_size) if crop else size))

    if crop:
        transform_list.append(transforms.RandomCrop(final_size))
    else:
        transform_list.append(transforms.Resize(final_size))

    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)    


def adaptive_instance_normalization(content_feat, style_feat):
    # [batch size, channels, h, w]
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_content_feat = (
        content_feat - content_mean.expand(size)
    ) / content_std.expand(size)

    return normalized_content_feat * style_std.expand(size) + style_mean.expand(size)


def calc_mean_std(feat, eps=1e-5):
    # [batch size, channels, h, w]
    size = feat.size()
    assert (len(size) == 4)

    batch_size, channels = size[:2]

    feat_mean = feat.view(batch_size, channels, -1).mean(dim=2).view(
        batch_size, channels, 1, 1
    )

    feat_var = feat.view(batch_size, channels, -1).var(
        dim=2, unbiased=False
    ) + eps

    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)

    return feat_mean, feat_std