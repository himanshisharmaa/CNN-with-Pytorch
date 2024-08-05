from torch.nn import Module
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.nn import Conv2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten

class LeNet(Module):
    def __init__(self,numChannels,classes):
        super(LeNet,self).__init__()

        # initialize first set of Conv -> Relu -> pool layers
        self.conv1=Conv2d(in_channels=numChannels,out_channels=20,kernel_size=(5,5))
        self.relu1=ReLU()
        self.maxPool1=MaxPool2d(kernel_size=(2,2),stride=(2,2))

        # initialize the second set of Conv -> ReLu -> Pool Layers
        self.conv2=Conv2d(in_channels=20,out_channels=50,kernel_size=(5,5))
        self.relu2=ReLU()
        self.maxPool2=MaxPool2d(kernel_size=(2,2),stride=(2,2))

        # initialize the first (and only)set of FC-> relu layers
        self.fc1=Linear(in_features=800,out_features=500)
        self.relu3=ReLU()

        #initialize our softmax classifier
        self.fc2=Linear(in_features=500,out_features=classes)
        self.logSoftmax=LogSoftmax(dim=1)
    
    '''
        the forward function serves a number of purposes:
        1. it connects layers/subnetworks together from variables defined
        in the constructor(i.e., __init__) of the class
        2. it defines the network architecture itself
        3. it allows the forward pass of the model to be performed, 
        resulting in our output predictions
        4. And thanks to Pytorch autgrad module, it allows us to perform 
        automatic differentiation and update our model weights
    '''

    def forward(self,x):
        # pass the input through our first set of CONV-> Relu->pool layer
        x=self.conv1(x)
        x=self.relu1(x)
        x=self.maxPool1(x)

        # pass the output from the previous layer through the second
        # set of Conv-> relu->pool layers
        x=self.conv2(x)
        x=self.relu2(x)
        x=self.maxPool2(x)

        # flatten the output from the previous layer ad pass it 
        # through our only set of fc -> relu layers
        x=flatten(x,1)
        x=self.fc1(x)
        x=self.relu3(x)

        # pass the output to our softmax classifier to get our output
        # predictions
        x=self.fc2(x)
        output=self.logSoftmax(x)

        return output 


