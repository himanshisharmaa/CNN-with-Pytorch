import numpy as np
np.random.seed(42)

from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
import argparse
import imutils
import torch
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to the trained PyTorch model")
args = vars(ap.parse_args())


device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the dtaset MNIST

testData=MNIST(root="data",train=False,download=True,transform=ToTensor())

idxs=np.random.choice(range(0,len(testData)),size=(10,))
testData=Subset(testData,idxs)

# initialize the test data loader
testDataLoader=DataLoader(testData,batch_size=1)

# load the model and set it to evaluation mode
model=torch.load(args["model"]).to(device)
model.eval()

with torch.no_grad():
    #loop over the test set
    for (image,label) in testDataLoader:
        #grab the original image and ground truth label
        orgImage=image.numpy().squeeze(axis=(0,1))
        gtLabel=testData.dataset.classes[label.numpy()[0]]

        # send the input to the device nd make predictions on it
        image=image.to(device)
        pred=model(image)


        # find the class label index with the largest corresponding
        #probability
        idx=pred.argmax(axis=1).cpu().numpy()[0]
        predLabel=testData.dataset.classes[idx]

        #convert the image from grayscale to RGB (so we can draw
        #on it) and resize it(so we can more easily see it on our
        # screen)
        orgImage=np.dstack([orgImage]*3)
        orgImage=imutils.resize(orgImage,width=128)
        # draw the predicted class label on it
        color=(0,255,0) if gtLabel == predLabel else (0,0,255)
        cv2.putText(orgImage,gtLabel,(2,25),cv2.FONT_HERSHEY_COMPLEX,0.95,
                    color,2)
        
        #display the result  in terminal and show the input image
        print("[INFO] ground truth label: {}, predicted label: {}".format(
			gtLabel, predLabel))
        cv2.imshow("image", orgImage)
        cv2.waitKey(0)
