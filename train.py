# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use('Agg')

from pyimg.lenet import LeNet
from sklearn.metrics import classification_report
from torch.utils.data import random_split,DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time

ap=argparse.ArgumentParser()
ap.add_argument("-m","--model",type=str,required=True,help="Path to output trained model")
ap.add_argument("-p","--plot",type=str,required=True,help="path to output loss/accuracy plot")
args=vars(ap.parse_args())

#define training hyperparameters
INIT_LR=1e-3
BATCH_SIZE=64
EPOCHS=10

# define the train and val splits
TRAIN_SPLIT=0.75
VAL_SPLIT=1-TRAIN_SPLIT

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device we are using is: {device}")

#load the MNIST dataset
print("[INFO] loading the MNIST Dataset....")
trainData=MNIST(root="data",train=True,download=True,transform=ToTensor())
testData=MNIST(root="data",train=False,download=True,transform=ToTensor())

# calculate the train/validation split
print("[INFO] generating the train/validation split....")
numTrainSamples=int(len(trainData)*TRAIN_SPLIT)
numValSamples=int(len(trainData)*VAL_SPLIT)

(trainData,valData)=random_split(trainData,[numTrainSamples,numValSamples],
                                 generator=torch.Generator().manual_seed(42))

#initialize the train, validation and test data loaders
trainDataloader=DataLoader(trainData,shuffle=True,batch_size=BATCH_SIZE)
valDataloader=DataLoader(valData,batch_size=BATCH_SIZE)
testDataloader=DataLoader(testData,batch_size=BATCH_SIZE)

# calculate steps per epoch for training and validation set
trainSteps=len(trainDataloader.dataset)//BATCH_SIZE
valSteps=len(valDataloader.dataset)//BATCH_SIZE


# initialize the lenet model
print("[INFO] initializing the LeNet Model....")
model=LeNet(
    numChannels=1,
    classes=len(trainData.dataset.classes)).to(device)

#initialize the optimizer and loss function
opt=Adam(model.parameters(),lr=INIT_LR)
lossFn=nn.NLLLoss()

#initialise a dictionary to store the training history
H={
    "train_loss":[],
    "train_acc":[],
    "val_loss":[],
    'val_Acc':[]

}

#measure how long training is going to take
print("[INFO] training the network..")
startTime=time.time()

#loop over our epochs
for e in range(0,EPOCHS):
    # set the model in training mode
    model.train()

    #initialize the total training and validation loss
    totalTrainLoss=0
    totalValLoss=0

    #initialize the number of correct predictions in the training and 
    # validation step
    trainCorrect=0
    valCorrect=0
    
    for (x,y) in trainDataloader:
        # send the input to the device
        (x,y)=(x.to(device),y.to(device))

        # perform a forward pass and calculate the training loss
        pred=model(x)
        loss=lossFn(pred,y)

        #zero out the gradient, perform the backpropagation and 
        # update the weights
        opt.zero_grad()
        loss.backward()
        opt.step()

        # add the loss to the total training loss so far and calculate
        # the number of correct predictions
        totalTrainLoss+=loss
        trainCorrect+=(pred.argmax(1)==y).type(torch.float).sum().item()

    # switch off the autograd for evaluation
    with torch.no_grad():
        model.eval()

        for (x,y) in valDataloader:
        # send the input to the device
            (x,y)=(x.to(device),y.to(device))

            # perform a forward pass and calculate the training loss
            pred=model(x)
            totalValLoss+=lossFn(pred,y)

            # calculate the number of correct predictions
            valCorrect+=(pred.argmax(1)==y).type(torch.float).sum().item()
    
    #Calculate the average training and validation accuracy
    avgTrainLoss=totalTrainLoss/trainSteps
    avgValLoss=totalValLoss/valSteps

    #calculate the training and validation accuracy
    trainCorrect=trainCorrect/len(trainDataloader.dataset)
    valCorrect=valCorrect/len(valDataloader.dataset)

    # update our training history

    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["train_acc"].append(trainCorrect)
    H["val_loss"].append(avgValLoss.cpu().detach().numpy())
    H["val_Acc"].append(valCorrect)

    #print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
		avgTrainLoss, trainCorrect))
    print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
		avgValLoss, valCorrect))


endTime=time.time()
print(f"[INFo] total time taken to train the model: {(endTime-startTime):.2f}s")


# we can now evaluate the network on the test set
print("[INFO] evaluating network...")
# turn off autograd for testing evaluation
with torch.no_grad():
	# set the model in evaluation mode
	model.eval()
	
	# initialize a list to store our predictions
	preds = []
	# loop over the test set
	for (x, y) in testDataloader:
		# send the input to the device
		x = x.to(device)
		# make the predictions and add them to the list
		pred = model(x)
		preds.extend(pred.argmax(axis=1).cpu().numpy())
# generate a classification report
print(classification_report(testData.targets.cpu().numpy(),
	np.array(preds), target_names=testData.classes))



plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="val_loss")
plt.plot(H["train_acc"], label="train_acc")
plt.plot(H["val_Acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
# serialize the model to disk
torch.save(model, args["model"])



