import torch
import torch.nn as nn
import torch.optim as optim

# 1. Dummy Data (100 samples, 10 features)
X_train = torch.randn(100, 10)
y_train = torch.randn(100, 1)

# 2. Simple Model
model = nn.Linear(in_features=10, out_features=1)

# 3. Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. The Training Loop
epochs = 5

for epoch in range(epochs):
    model.train() # Set model to training mode
    
    # Forward pass: compute predictions
    predictions = model(X_train)
    
    # Compute the loss
    loss = criterion(predictions, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad() # Clear gradients from the previous step
    loss.backward()       # Compute new gradients (backpropagation)
    optimizer.step()      # Update model weights
    
    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")