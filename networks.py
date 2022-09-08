import torch
import torch.nn as nn
import torch.nn.functional as F


class old_Z(nn.Module):
    """
    Create and run the encoder network.
    """

    def __init__(self, HP):
        super(old_Z, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, 2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)

        self.conv_exp = nn.Conv2d(1, 32, 3, padding=1)

        self.fc1 = nn.Linear(32 * 17 * 17, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)
        self.fc5 = nn.Linear(10, HP['Z_dim'])
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        :param x: Image
        :return: The encoder value of the image
        """
        x = x.view(-1, 2, 100, 100)

        x = self.relu(self.conv1(x))
        x = F.max_pool2d(self.relu(self.conv2(x)), 2)
        x = F.max_pool2d(self.relu(self.conv3(x)), 3)
        x = x.view(-1, 32 * 17 * 17)

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = self.fc5(x)
        return x


class fc_Z(nn.Module):
    def __init__(self, HP):
        super(fc_Z, self).__init__()
        self.HP = HP
        self.fc1 = nn.Linear(126, 100)
        self.fc2 = nn.Linear(100, 70)
        self.fc3 = nn.Linear(70, 30)
        self.fc4 = nn.Linear(30, 4)
        self.fc5 = nn.Linear(4, 1)

    def forward(self, x):
        x = x.view(-1, 126)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = self.fc5(x)
        return x

class T(nn.Module):
    """
    Create and run the transition network.
    """

    def __init__(self, HP):
        super(T, self).__init__()
        self.HP = HP

        self.fc1 = nn.Linear(self.HP['Z_dim'], 10)
        self.fc2 = nn.Linear(10, 30)
        self.fc3 = nn.Linear(30, 30)
        self.fc4 = nn.Linear(30, 4)
        self.fc5 = nn.Linear(4, self.HP['Z_dim'])

        self.const = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        """
        :param x: The encoder value of an image
        :return: The transition two the value of the next image in the sequence
        """
        x = x.view(-1, self.HP['Z_dim'])
        x_in = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = self.fc5(x) + x_in

        return x
