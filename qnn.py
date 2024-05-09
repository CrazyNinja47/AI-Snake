import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from IPython import display


class QNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size): 
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QLearning:
    def __init__(self, input_size, hidden_size, output_size, gamma, lr=.001):
        self.qnn = QNN(input_size, hidden_size, output_size)
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(self.qnn.parameters(), lr=self.lr) 
        self.loss_function = nn.MSELoss() 
        self.memory = []

    def train(self, state, action, reward, next_state, done): 
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        pred = self.model(state)

        new_pred = pred.clone() 
        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]:
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))
            new_pred[i][torch.argmax(action[i]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.loss_function(new_pred, pred)
        loss.backward()
        self.optimizer.step()

    def next_move(self, state):
        state = torch.FloatTensor(state)
        with torch.no_grad():
            q_values = self.qnn(state)
        move = q_values.argmax().item()
        return move

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


