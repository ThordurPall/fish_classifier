from torch import nn
import torch.nn.functional as F

class NeuralNetworkModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = 12288
        self.output = 9
        # Define fully connected layers
        self.fc1 = nn.Linear(self.input, int(self.input/3))
        self.fc2 = nn.Linear(int(self.input/3), int(self.input/(3*2)))
        self.fc3 = nn.Linear(int(self.input/(3*2)), int(self.input/(3*2*2)))
        self.fc4 = nn.Linear(int(self.input/(3*2*2)), int(self.input/(3*2*2*2)))
        self.fc5 = nn.Linear(int(self.input/(3*2*2*2)), int(self.input/(3*2*2*2*2)))
        self.fc6 = nn.Linear(int(self.input/(3*2*2*2*2)), int(self.input/(3*2*2*2*2*2)))
        self.fc7 = nn.Linear(int(self.input/(3*2*2*2*2*2)),self.output)

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        """ Forward pass through the network, returns the output logits """

        # Flattening input tensor except for the minibatch dimension
        x = x.view(x.shape[0], -1)

        # Fully connected layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.dropout(F.relu(self.fc5(x)))
        x = self.dropout(F.relu(self.fc6(x)))

        # Output so no dropout here
        x = F.log_softmax(self.fc7(x), dim=1)
        return x