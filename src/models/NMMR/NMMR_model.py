import torch
import torch.nn as nn


class cnn_mono_for_dsprite(nn.Module):
    def __init__(self, train_params):
        super(cnn_mono_for_dsprite, self).__init__()
        self.train_params = train_params
        self.batch_size = train_params["batch_size"]

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # A blocks
        self.conv_A_0 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding='same')
        self.conv_A_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.conv_A_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.conv_A_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same')
        self.projection_A = nn.Linear(128*16*16, 128)

        # W blocks
        self.conv_W_0 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding='same')
        self.conv_W_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.conv_W_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.conv_W_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same')
        self.projection_W = nn.Linear(128*16*16, 128)

        # MLP
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, A, W):
        # A head
        A = torch.relu(self.conv_A_0(A))
        A = torch.relu(self.conv_A_1(A))
        A = self.max_pool(A)
        A = torch.relu(self.conv_A_2(A))
        A = torch.relu(self.conv_A_3(A))
        A = self.max_pool(A)
        A = torch.flatten(A, start_dim=1)
        A = self.projection_A(A)

        # W head
        W = torch.relu(self.conv_W_0(W))
        W = torch.relu(self.conv_W_1(W))
        W = self.max_pool(W)
        W = torch.relu(self.conv_W_2(W))
        W = torch.relu(self.conv_W_3(W))
        W = self.max_pool(W)
        W = torch.flatten(W, start_dim=1)
        W = self.projection_W(W)

        # MLP
        x = torch.cat((A, W), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x
