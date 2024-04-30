import torch.nn as nn
import random
import torch
from torch.utils.data import Dataset, DataLoader

# One issue to consider is the tail. It will grow throughout the game, so
# we could have a max size, and pad missing tail segments with zeros?
max_length = 20  # temp
# So we have the size of each snake's tail, *2 for (x, y), plus food (x, y), plus head (x, y) * 2.

input_size = (max_length + max_length) * 2 + 2 + (2 * 2)
hidden_size = 256  # Can be tuned
output_size = 3  # left, straight, right

# "up" -> 0
# "down" -> 1
# "left" -> 2
# "right" -> 3


class MyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# input a bunch of game states for training
data = None  # get_state_representation(game_state)

targets = [0, 1, 2, 3]
dataset = MyDataset(data, targets)

batch_size = 32  # can be tuned
shuffle = True  # Whether to shuffle the data
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class SnakeNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SnakeNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        output_probs = nn.functional.softmax(x, dim=1)
        return output_probs

    def train(model, optimizer, criterion, memory, batch_size):
        # Set model to training mode
        model.train()
        num_epochs = 1000

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


def get_state_representation(game_state):
    # Convert game state to tensor
    pass


def select_action(state, model):
    # Return a direction to snakebattle.py
    pass
