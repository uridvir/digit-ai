from train import Net
import torch
from torchvision import transforms
from matplotlib import pyplot as plt

import cv2 as cv

"""Import the neural network"""
net = Net()
net.load_state_dict(torch.load("net.pth"))


"""Get input from the webcam"""
print("Getting camera...")

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Got camera!")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = ~cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord(' '):
        """Feed frame into neural net"""

        image = cv.resize(gray, (28, 28))
        plt.imshow(image)
        plt.show()
        data = transforms.ToTensor()(image)
        response = input("Use this image (y/n)?")
        if response == "y":
            print("Net result is ", torch.max(net(data), 1)[1][0].item())
            plt.imshow(image)
            plt.show()
        plt.close()

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
