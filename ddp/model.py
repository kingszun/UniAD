# model.py
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms

def find_project_root(current_path, project_name='ad-e2e-model-embedded'):
    while True:
        if os.path.basename(current_path) == project_name:
            return current_path
        parent = os.path.dirname(current_path)
        if parent == current_path:
            raise FileNotFoundError(f"Project root '{project_name}' not found")
        current_path = parent

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

def load_mnist_data(data_dir):    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    return dataset