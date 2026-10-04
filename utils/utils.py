from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms


class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        super(ImageFolderDataset, self).__init__()

        self.root = root
        self.transform = transform

        self.files = [
            p for p in os.listdir(root)
            if p.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        MAX_PIXELS = 50_000_000

        valid_files = []

        for p in self.files:
            image_path = os.path.join(self.root, p)

            try:
                # First check image size
                with Image.open(image_path) as img:
                    width, height = img.size
                    pixels = width * height

                    if pixels > MAX_PIXELS:
                        print(
                            f"Skipping huge image: {p} "
                            f"({width} x {height} = {pixels:,} pixels)"
                        )
                        continue

                    # Check whether the image is actually valid
                    img.verify()

                # Re-open after verify()
                with Image.open(image_path) as img:
                    img.load()

                valid_files.append(p)

            except Exception as e:
                print(f"Skipping invalid image: {p} | {e}")

        self.files = valid_files

        print(f"Loaded {len(self.files)} valid images from {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.files[idx])

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

def get_transform(size, crop, final_size):
    transform_list = []
    if size > 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.RandomCrop(size))
    else:
        transform_list.append(transforms.Resize(final_size))

    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)  # to store above steps as a single pipeline

def adaptive_instance_normalization(content_feat, style_feat):
    # [batch size, channels, h, w]
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_content_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_content_feat * style_std.expand(size) + style_mean.expand(size)

def calc_mean_std(feat, eps=1e-5):
    # [batch size, channels, h, w]
    size = feat.size()
    assert (len(size) == 4) # make sure input given is 4 dimensional
    batch_size, channels = size[:2]
    feat_mean = feat.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1) # here making as 3d using as 2d and again making as 4d for feature use
    feat_var = feat.view(batch_size, channels, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)
    return feat_mean, feat_std