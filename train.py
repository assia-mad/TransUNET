import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import BreastCancerDataset, TransUNet
import torch.nn as nn

def train_model(model, seg_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0

        for images, labels in seg_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)
            print("loss",loss)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(seg_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    print('Finished Training')