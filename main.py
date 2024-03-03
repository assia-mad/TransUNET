import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import BreastCancerDataset, TransUNet
from train import train_model
import torch.nn as nn

def main():
    # Dataset setup
    images_dir = 'C:/Users/windows/Desktop/Datasets/BC_dataset/seg/train/images'
    labels_dir = 'C:/Users/windows/Desktop/Datasets/BC_dataset/seg/train/labels'
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    seg_dataset = BreastCancerDataset(images_dir=images_dir, labels_dir=labels_dir, transform=transform)
    seg_loader = DataLoader(seg_dataset, batch_size=2, shuffle=True)

    # Model setup
    model = TransUNet(in_channels=1, out_channels=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training parameters
    num_epochs = 10

    # Start training
    print("we will start training")
    train_model(model, seg_loader, criterion, optimizer, num_epochs, device)

    # Save the trained model
    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    main()
