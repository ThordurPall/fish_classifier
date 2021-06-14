import torch.nn.functional as F
from torch import nn


class NeuralNetworkModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define fully connected layers
        self.fc1 = nn.Linear(49152, 16384)
        self.fc2 = nn.Linear(16384, 8192)
        self.fc3 = nn.Linear(8192, 4096)
        self.fc4 = nn.Linear(4096, 2048)
        self.fc5 = nn.Linear(2048, 1024)
        self.fc6 = nn.Linear(1024, 512)
        self.fc7 = nn.Linear(512, 128)
        self.fc8 = nn.Linear(128,9)

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        """ Forward pass through the network, returns the output logits """

        # Flattening input tensor except for the minibatch dimension
        x = x.view(x.shape[0], -1)
        print(x.shape)

        # Fully connected layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.dropout(F.relu(self.fc5(x)))
        x = self.dropout(F.relu(self.fc6(x)))
        x = self.dropout(F.relu(self.fc7(x)))
 
        # Output so no dropout here
        x = F.log_softmax(self.fc8(x), dim=1)
        return x