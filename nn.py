import torch.nn as nn
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np


# Used
    # data = [*gs.snake_eyes(1), move, 'None' if gs.winner is None else 'true']

# with this
    # def snake_eyes(self, player):
    #     if player == 1:
    #         target = self.player1
    #         enemy = self.player2
    #     else:
    #         target = self.player2
    #         enemy = self.player1
    #     result = []
    #     # Tell snake which way to turn for food one hot-encoded
    #     #   0     1     2
    #     # Left Center Right

    #     # NOTE NOT SURE ABOUT THIS!

    #     # This returns which side the food is on, one hot encoded.
    #     # 0 LEFT  1 FORWARD  2 RIGHT

    #     self.append_food_side(player,result)
    #     # Get distance from food, elucidian
    #     result.append( math.sqrt( ((target.x - self.food[0]) ** 2) + ((target.y - self.food[1]) ** 2)))
    #     # else:
    #         #sometimes food doesn't exist, don't want to throw an error with a None
    #         # No food, go right
    #         # result.append(0)
    #         # result.append(0)
    #         # result.append(1)

    #         # # 0 distance
    #         # result.append(0)

    #     # Wall distances - top and left
    #     result.append(target.x)
    #     result.append(target.y)

    #     # Wall distances - bottom and right
    #     result.append(abs(MAP_SIZE[0] - target.x))
    #     result.append(abs(MAP_SIZE[1] - target.y))

    #     # Vision cone didn't work well.  New plan:
    #     #   Go in a line in front of snake "vision"
    #     #   Return distance to deadly object (snake, wall)
    #     #   Check each cardinal direction (left, forward, right)

    #     left = self.get_left(player)
    #     right = self.get_right(player)
    #     spot = 0
    #     step = 0
    #     while spot != 1:
    #         step += 1
    #         spot = self.get_square(player,(target.x + (left[0] * step)),(target.y + (left[1]* step)) )
    #     result.append(step)

    #     # center:
    #     spot = 0
    #     step = 0
    #     while spot != 1:
    #         step += 1
    #         spot = self.get_square(player,(target.x + (target.direction[0] * step)),(target.y + (target.direction[1] * step)) )
    #     result.append(step)

    #     #right
    #     spot = 0
    #     step = 0
    #     while spot != 1:
    #         step += 1
    #         spot = self.get_square(player,(target.x + (right[0] * step)),(target.y + (right[1]* step)) )
    #     result.append(step)
    #     #Enemy Distance to head
    #     result.append( math.sqrt( ((target.x - enemy.x) ** 2) + ((target.y - enemy.y) ** 2)))
    #     return result





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
# print(f"Y_move{df.iloc[:, -2]}")

y_winner = df.iloc[:, -1].apply(lambda x: 1 if x == 'true' else 0).values  # Convert winner indicator to binary
# X = np.concatenate((coordinates, df.iloc[:, 4:-2].values), axis=1)
X = df.iloc[:,:-2].values

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X)
y_move_tensor = torch.tensor(y_move, dtype=torch.long)
# print(f"Y_move!!{y_move}")
# print(f"Y_move tensor!! {y_move_tensor}")
y_winner_tensor = torch.tensor(y_winner)
input_size = X_tensor.shape[1]
# print(f"input size: {input_size}")


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
# print(f"X_tensor {X_tensor}")
# print(f"y_move_tensor printer{y_move_tensor}")
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


class SnakeNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):

        super(SnakeNN, self).__init__()
        self.finput = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc1_1= nn.Linear(hidden_size, hidden_size)
        self.fc1_2 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc2_3 = nn.Linear(hidden_size * 2 , hidden_size * 3)
        self.fc3_2 = nn.Linear(hidden_size * 3, hidden_size * 2)
        self.fc2_1 = nn.Linear(hidden_size * 2, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(.05)
        self.batchNorm1 = nn.BatchNorm1d(hidden_size)
        self.batchNorm2 = nn.BatchNorm1d(hidden_size * 2)
        self.batchNorm3 = nn.BatchNorm1d(hidden_size * 3)

    def forward(self, x):
        x = x.float()
        x = self.finput(x)
        # x = self.batchNorm1(x)
        x = self.relu(x)
        # These are just 1 hidden -> 1 hidden NNs, I made 2* and 3* to mess with.
        x = self.fc1_1(x)
        x = self.relu(x)
        x = self.fc1_1(x)
        x = self.relu(x)
        x = self.fc1_1(x)
        x = self.relu(x)
        x = self.fc1_1(x)
        x = self.relu(x)

        # x = self.dropout(x)
        # x = self.fc2_1(x)
        # x = self.relu(x)
        # x = self.fc1_1(x)
        # x = self.relu(x)
        x = self.output(x)



        # print(f"X is now: {x}")
        # output_probs = nn.functional.softmax(x, dim=1)
        output_probs = x
        return output_probs

model = SnakeNN(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0006)
# 0.0006 - .34
# optimizer = optim.SGD(model.parameters(), lr=.001, momentum=0.9)
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

            # print(f"Outputs: {outputs}")
            # print(f"targets: {targets}")
            loss = criterion(outputs, targets)
            # print(f"loss: {loss}")

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



def read_file_and_sample_lines(filename):
    all_lines = []
    
    with open(filename, 'r') as file:
        for line in file:
            # Convert line to list, remove the last two elements, and ensure all are floats
            numeric_line = [float(x) if x.isdigit() else 0 for x in line.strip().split(',')[:-2]]
            all_lines.append(numeric_line)
    
    if len(all_lines) >= 5:
        random_lines = random.sample(all_lines, 5)
    else:
        random_lines = all_lines
    
    return random_lines

# Assuming your model, criterion, and optimizer are properly defined as before
filename = 'game_data.csv'
input_data = read_file_and_sample_lines(filename)
input_tensor = torch.tensor(input_data, dtype=torch.float32)

model.eval()  # Make sure model is in evaluation mode

with torch.no_grad():
    predicted_outputs = model(input_tensor)

predicted_moves = torch.argmax(predicted_outputs, dim=1)
move_labels = {0: "LEFT", 1: "STRAIGHT", 2: "RIGHT"}
predicted_moves = [move_labels[move.item()] for move in predicted_moves]

#predicted moves according to the random positions selected and the probability tensor from softmax

for i in predicted_moves:
    print(i)

print(predicted_outputs)