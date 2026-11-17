import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__() # Always call the parent class init
        
        # 1. Feature Extraction Layers
        # Input: 1 channel (grayscale). Output: 16 channels. 3x3 kernel.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Input: 16 channels. Output: 32 channels.
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # 2. Classification Layers
        # A 28x28 image pooled twice by 2x2 becomes 7x7. 
        # Flattened shape: 32 channels * 7 height * 7 width = 1568
        self.fc1 = nn.Linear(in_features=1568, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10) # 10 output classes

    def forward(self, x):
        # Pass through Conv1 -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv1(x)))
        
        # Pass through Conv2 -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the tensor (keep the batch dimension at index 0)
        x = torch.flatten(x, start_dim=1)
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 
        # Note: We omit Softmax here because PyTorch's CrossEntropyLoss applies it internally
        
        return x

# Quick test to verify shapes
model = SimpleCNN()
dummy_batch = torch.randn(4, 1, 28, 28) # (Batch_size, Channels, Height, Width)
predictions = model(dummy_batch)

print(f"Output shape: {predictions.shape}") # Expected: [4, 10]