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
    image = ~cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    alpha = 1.5 # Contrast control (1.0-3.0)
    beta = 0 # Brightness control (0-100)
    image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)

    _, image = cv.threshold(image, 200, 255, cv.THRESH_TOZERO)

    # Crop center third of image
    w, h = image.shape
    image = image[int(w/3):int(2*w/3), int(h/3):int(2*h/3)]

    # Display the resulting frame
    cv.imshow('frame', image)
    if cv.waitKey(1) == ord(' '):
        """Feed frame into neural net"""

        image = cv.resize(image, (28, 28))
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
