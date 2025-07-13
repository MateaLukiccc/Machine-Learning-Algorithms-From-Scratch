from torch import nn

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=(1,1))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1))
        
        # block 2
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(1,1))
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1))
        
        # block 3
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=(1,1))
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1))
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1))
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1))
        
        # block 4
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=(1,1))
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1))
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1))
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1))
        
        # block 5
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1))
        self.conv14 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1))
        self.conv15 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1))
        self.conv16 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1))
        
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.fc1 = nn.Linear(4096, 4096)
        self.fc1 = nn.Linear(4096, 4096)
        self.fc1 = nn.Linear(4096, 1000)
        
    