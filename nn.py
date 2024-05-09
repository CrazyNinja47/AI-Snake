import torch.nn as nn
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np


# One issue to consider is the tail. It will grow throughout the game, so
# we could have a max size, and pad missing tail segments with zeros?
max_length = 20  # temp
# So we have the size of each snake's tail, *2 for (x, y), plus food (x, y), plus head (x, y) * 2.


hidden_size = 256  # Can be tuned
output_size = 3  # left, straight, right

# "up" -> 0
# "down" -> 1
# "left" -> 2
# "right" -> 3


import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
scaler = StandardScaler()

# Load data assuming no headers, so we specify header=None
df = pd.read_csv('game_data.csv', header=None)

p1_coords = df.iloc[:, :2].values  # First two columns
p2_coords = df.iloc[:, 2:4].values
coordinates = np.concatenate((p1_coords, p2_coords), axis=1)


y_move = df.iloc[:, -2].apply(lambda x: 0 if x == 'LEFT' else (1 if x == 'STRAIGHT' else 2)).values  # Convert directions to numerical values
y_winner = df.iloc[:, -1].apply(lambda x: 1 if x == 'true' else 0).values  # Convert winner indicator to binary
X = np.concatenate((coordinates, df.iloc[:, 4:-2].values), axis=1)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X)
y_move_tensor = torch.tensor(y_move, dtype=torch.long)
y_winner_tensor = torch.tensor(y_winner) 
input_size = X_tensor.shape[1]


print(X_tensor)
print(y_move_tensor)
print(y_winner_tensor)

# Define dataset
class GameDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create dataset and dataloader
dataset = GameDataset(X_tensor, y_move_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


class SnakeNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SnakeNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.float() 
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        output_probs = nn.functional.softmax(x, dim=1)
        return output_probs
    
model = SnakeNN(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, optimizer, criterion, batch_size):
    # Set model to training mode
    model.train()
    num_epochs = 100

    for epoch in range(num_epochs):
        running_loss = 0.0

        # Iterate over the training dataset
        for inputs, targets in train_loader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Accumulate the loss
            running_loss += loss.item() * inputs.size(0)

        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)

        # Print epoch statistics
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    print("Training complete")

train(model, optimizer, criterion, train_loader)

#Testing the NN giving it a state and lookin at the output 

# Assume game_state is the input game state in the same format as your training data
# Convert the game state to a tensor

game_state_tensor = torch.tensor(game_state)

# Pass the game state tensor through the model
with torch.no_grad():
    model.eval()  # Set the model to evaluation mode
    output_probs = model(game_state_tensor)

# Interpret the output
# For example, you can get the predicted move or winner based on the output_probs
predicted_move = torch.argmax(output_probs, dim=1).item()