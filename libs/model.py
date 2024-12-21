from torch import nn


class ImageRepairingCNN(nn.Module):
    """
    A sample PyTorch CNN model
    """
    def __init__(self, input_shape=(3, 600, 400)):
        super(ImageRepairingCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=30, kernel_size=(9, 9), padding=(4, 4))
        self.conv2 = nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(9, 9), padding=(4, 4))
        self.conv3 = nn.Conv2d(in_channels=30, out_channels=input_shape[0], kernel_size=(9, 9), padding=(4, 4))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x
