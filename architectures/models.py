from torch import nn
from torch.nn import functional as F


class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1024)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

class LeNet5(nn.Module):          
     
    def __init__(self):     
        super(LeNet5, self).__init__()
        # Convolution (In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2, bias=True)
        # Max-pooling
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)
        # Convolution
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, bias=True)
        # Max-pooling
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2) 
        # Fully connected layer
        self.fc1 = nn.Linear(64*5*5, 1024)   # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
        self.fc2 = nn.Linear(1024, 10)       # convert matrix with 120 features to a matrix of 84 features (columns)
        #self.fc3 = nn.Linear(84, 10)        # convert matrix with 84 features to a matrix of 10 features (columns)
        
    def forward(self, x):
        # convolve, then perform ReLU non-linearity
        #print(x.shape)
        x = nn.functional.relu(self.conv1(x))  
        #print(x.shape)
        # max-pooling with 2x2 grid 
        x = self.max_pool_1(x) 
        #print(x.shape)
        # convolve, then perform ReLU non-linearity
        x = nn.functional.relu(self.conv2(x))
        #print(x.shape)
        # max-pooling with 2x2 grid
        x = self.max_pool_2(x)
        print(x.shape)
        # first flatten 'max_pool_2_out' to contain 16*5*5 columns
        # read through https://stackoverflow.com/a/42482819/7551231
        x = x.view(-1, 64*5*5)
        #print(x.shape)
        # FC-1, then perform ReLU non-linearity
        x = nn.functional.relu(self.fc1(x))
        #print(x.shape)
        # FC-2, then perform ReLU non-linearity
        #x = nn.functional.relu(self.fc2(x))
        # FC-3
        x = self.fc2(x)
        
        return x