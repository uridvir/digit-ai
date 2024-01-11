import torch
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class TrainDataset(Dataset):
    """Custom dataset type for reading the training CSV"""

    def __init__(self, transforms=None):
        print("Reading training data...")
        self.training_frame = pd.read_csv("train.csv")
        print("Done!")
        self.transforms = transforms

    def __len__(self):
        return len(self.training_frame)

    def __getitem__(self, i):
        array = np.array(self.training_frame.iloc[i,1:]).reshape(28, 28)
        image = Image.fromarray(array.astype(np.uint8))

        if self.transforms:
            image = self.transforms(image)

        label = self.training_frame.iloc[i,0]
        return image, label

class Net(nn.Module):
    """The neural network"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train():
    """Get the training data"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    train_dataset = TrainDataset(transforms=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    """Create the neural network"""
    net = Net()

    # Instantiate the model and move it to the appropriate device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net.to(device)

    """Train on the data"""
    print("Begin training...")

    # Loss Function
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)

    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # epochs = 8
    epochs = 8

    for epoch in range(epochs):

        print(f"Epoch {epoch + 1}\n-------------------------------")

        size = len(train_loader.dataset)
        num_batches = len(train_loader)
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch, (X, y) in enumerate(train_loader):
            # Move data to the device (GPU or CPU)
            X, y = X.to(device), y.to(device)

            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            # Print statistics every 100 batches
            if batch % 100 == 0:
                loss_current = running_loss / (batch + 1)
                accuracy_current = 100 * correct / total
                print(f"[Epoch {epoch + 1}, Batch {batch + 1:5d}] loss: {loss_current:.4f}, Accuracy: {accuracy_current:.2f}%")

        # Average loss and accuracy for the epoch
        epoch_loss = running_loss / num_batches
        epoch_acc = 100 * correct / size
        print(f"End of Epoch {epoch + 1}: Avg loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%\n")

        scheduler.step()  # Update learning rate after completing an epoch

    print("Done training!")

    """Save the neural network to disk"""
    torch.save(net.state_dict(), "net.pth")

if __name__ == "__main__":
    train()
