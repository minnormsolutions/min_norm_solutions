# Shared Neural Nets used by NMMR and Naive Net.
# mlp_general is used for most data, cnn/mlp_for_dsprite is used only for dSprites.

import torch
import torch.nn as nn


class mlp_general(nn.Module):
    
    def __init__(self, input_dim, train_params):
        super(mlp_general, self).__init__()

        self.train_params = train_params
        self.network_width = train_params["network_width"]
        self.network_depth = train_params["network_depth"]

        self.layer_list = nn.ModuleList()
        for i in range(self.network_depth):
            if i == 0:
                self.layer_list.append(nn.Linear(input_dim, self.network_width))
            else:
                self.layer_list.append(nn.Linear(self.network_width, self.network_width))
        self.layer_list.append(nn.Linear(self.network_width, 1))

    def forward(self, x):
        for ix, layer in enumerate(self.layer_list):
            if ix == (self.network_depth + 1):  # if last layer, don't apply relu activation
                x = layer(x)
            else:
                x = torch.relu(layer(x))

        return x

class cnn_for_dsprite(nn.Module):
    """CNN to extract hidden parameters from image data."""
    
    def __init__(self, train_params):
        super(cnn_for_dsprite, self).__init__()
        self.train_params = train_params
        self.batch_size = train_params["batch_size"]

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_0 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding='same')
        self.conv_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.conv_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same')
        self.projection = nn.Linear(128*16*16, 128)
    
    def forward(self, x):
        x = torch.relu(self.conv_0(x))
        x = torch.relu(self.conv_1(x))
        x = self.max_pool(x)
        x = torch.relu(self.conv_2(x))
        x = torch.relu(self.conv_3(x))
        x = self.max_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.projection(x)

        return x

class mlp_for_dsprite(nn.Module):
    
    def __init__(self, input_dim, train_params):
        super(mlp_for_dsprite, self).__init__()
        self.train_params = train_params
        self.batch_size = train_params["batch_size"]

        # input_dim = 128 (A) + 128 (W) + 3 (Z),
        # depending on which of A, W, Z are acually used

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x
