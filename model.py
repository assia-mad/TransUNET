
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import os


class BreastCancerDataset(Dataset):
        def __init__(self, images_dir, labels_dir, transform=None):
            self.images_dir = images_dir
            self.labels_dir = labels_dir
            self.transform = transform
            self.images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_dir) for f in filenames if os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg']]
            self.labels = [os.path.join(dp, f) for dp, dn, filenames in os.walk(labels_dir) for f in filenames if os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg']]

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_name = self.images[idx]
            label_name = self.labels[idx]
            image = Image.open(img_name).convert('L')  # Convert to grayscale
            label = Image.open(label_name).convert('L')
            if self.transform:
                image = self.transform(image)
                label = self.transform(label)
            return image, label

class TransUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransUNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        #  transformer layer
        transformer_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dropout=0.1)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder forward pass
        x = self.encoder(x)

        # Save the original shape for later
        original_shape = x.shape

        # Reshape for transformer input (N, C, H*W) -> (H*W, N, C)
        x = x.view(x.size(0), x.size(1), -1).permute(2, 0, 1)

        # Transformer forward pass
        x = self.transformer(x)

        # Reshape back to the original shape before the transformer
        x = x.permute(1, 2, 0).view(original_shape[0], original_shape[1], original_shape[2], original_shape[3])

        # Decoder forward pass
        x = self.decoder(x)

        return x


