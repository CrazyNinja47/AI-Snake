import torch.nn as nn

# One issue to consider is the tail. It will grow throughout the game, so
# we could have a max size, and pad missing tail segments with zeros?
max_length = 20  # temp
# So we have the size of each snake's tail, *2 for (x, y), plus food (x, y), plus head (x, y) * 2.

input_size = (max_length + max_length) * 2 + 2 + (2 * 2)
hidden_size = 256  # Can be tuned
output_size = 3  # left, straight, right


class SnakeNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SnakeNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.network(x)


def get_state_representation(game_state):
    # Convert game state to tensor
    pass


def select_action(state, model):
    # Return a direction to snakebattle.py
    pass


def train(model, optimizer, criterion, memory, batch_size):
    pass
