# Collision Avoidance - Live Demo

'''
In this program, we'll use the model we trained to detect whether the
robot is `free` or `blocked` to enable a collicion avoidance behavior 
on the robot.
'''

'''
First, we import all packages we need.
'''
import torch
import torchvision
import torch.nn.functional as F
import cv2
import numpy as np
import time

from jetbot import Robot, Camera

'''
Load the trained model

We'll assume that you've already had `best_model.pth`. Now, you should
initialize the PyTorch model and load the trained wights from 
`best_model.pth`.
'''

model = torchvision.models.alexnet(pretrained=False)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)

model.load_state_dict(torch.load('best_model.pth'))

'''
Similarly, we use CPU to run the model.
'''
device = torch.device('cpu')
model = model.to(device).eval()

'''
Create the preprocessing function

The format that we trained our model doesn't exactly match the format 
of the camera. To do that, we need to do some preprocessing. This involves
the following steps:
	1. Convert from BGR to RGB. 
	2. Convert from HWC layout to CHW layout.
	3. Normalize using same parameters as we did during training. Our camera
	   provides in [0,255] range and training loaded images in [0,1] range, 
	   so we need to scale by 255.0.
	4. Transfer the data from CPU memory to the device you choose.
	5. Add a batch dimension.
'''
mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])

normalize = torchvision.transforms.Normalize(mean, stdev)

def preprocess(camera_value):
	global device, normalize
	x = camera_value
	x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
	x = x.transpose((2, 0, 1))
	x = torch.from_numpy(x).float()
	x = normalize(x)
	x = x.to(device)
	x = x[None, ...]
	return x

'''
Also, we need to initialize the robot and its camera, and define the 
base speed at 20%.
'''
robot = Robot()
camera = Camera.instance(width=224, height=224)
speed = 0.2

'''
Now, we can create an infinity loop for doing the routine of the robot.
The robot's routine of collision avoidance should do the following steps:
	1. Pre-process the camera image.
	2. Execute the neural network.
	3. While the neural network output indicates we're blocked, we'll turn 
	   left, otherwise, we go forward.
'''
while True:

	'''
	We copy the image from camera since we want to draw some information
	on the image and do not want to change the original one.
	'''
	image = camera.value.copy()

	x = preprocess(image)
	y = model(x)

	'''
	We apply the `softmax` function to normalize the output vector, so it 
	sums to 1 (which makes it a probability distribution).
	'''
	y = F.softmax(y, dim=1)

	prob_blocked = float(y.flatten()[0])

	if (prob_blocked < 0.5):
		cv2.putText(image, 'Free', (20, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
		robot.forward(speed)
	else:
		cv2.putText(image, 'Blocked', (20, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
		robot.left(speed)

	cv2.imshow('camera', image)
	key = cv2.waitKey(1)

	'''
	We can press the space (0x20) key to leave the loop.
	'''
	if (key == 0x20):
		break

camera.stop()
cv2.destroyAllWindows()