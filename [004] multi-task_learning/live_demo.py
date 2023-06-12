# Multi-task Learning - Live Demo

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

We'll assume that you've already had `best_multi_task_model_xy.pth`. Now, 
you should initialize the PyTorch model and load the trained wights from 
`best_multi_task_model_xy.pth`.
'''

model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 8)

model.load_state_dict(torch.load('best_multi_task_model_xy.pth'))

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
Also, we need to initialize the robot and its camera.
'''
robot = Robot()
camera = Camera.instance(width=224, height=224)

'''
Now, we'll define some parameters to control JetBot:
	1. Speed Control (speed): Set a value to start your JetBot. Here,
	   we've set a default value to 20%.
	2. Steering Gain Control (steering_gain and steering_dgain): If you see 
	   JetBot is wobbling, you need to reduce "steering_gain" or "steering_dgain" 
	   till it is smooth.
	3. Steering Bias Control (steering_bias): If you see JetBot is biased
	   toward extreme right or extreme left side of the track, you should
	   control this variable till JetBot start following line or track in
	   the center. This accounts for motor biases as well as camera offsets.
'''
speed = 0.2
steering_gain  = 0.045
steering_dgain = 0.01
steering_bias = 0.0

'''
Next, we can create an infinity loop for doing the routine of the robot.
The routine will do the following steps:
	1. Pre-process the camera image.
	2. Execute the neural network.
	3. Compute the approximate steering value.
	4. Get the probability of blocked image.
	5. Get the classification result of traffic light color.
	5. If the probability of blocked image is less than 0.5 or the traffic light
	   color is red or yellow, motors should stop. Otherwise, control the motors 
	   using proportional / derivative control (PD control)
'''

'''
To visualize the steering angle, we can define a function to draw it on the
image.
'''
def draw_steering(image, angle, color=(0, 255, 0), radius=30):

	center = (image.shape[1]//2, image.shape[0] - 1)
	theta = angle - 3.14 / 2.0

	rcos = radius * np.cos(theta)
	rsin = radius * np.sin(theta)

	pt1 = (int(0.6 * rcos + center[0]), int(0.6 * rsin + center[1]))
	pt2 = (int(1.4 * rcos + center[0]), int(1.4 * rsin + center[1]))

	cv2.circle(image, center, radius, color, 5)
	cv2.line(image, pt1, pt2, color, 5)

angle_last = 0.0

while True:

	image = camera.value.copy()

	'''
	Split the model outputs.
	'''
	outputs = torch.split(model(preprocess(image)), [ 2, 2, 4 ], dim=1)

	xy = outputs[0].detach().float().cpu().numpy().flatten()

	x = xy[0]
	y = (0.5 - xy[1]) / 2.0

	angle = np.arctan2(x, y)
	pid = angle * steering_gain + (angle - angle_last) * steering_dgain
	angle_last = angle

	steering = pid + steering_bias

	'''
	Compute the probability of blocked image.
	'''
	p = F.softmax(outputs[1], dim=1)
	prob_blocked = float(p.flatten()[1])

	'''
	Get the traffic light color classification result.
	0: red
	1: yellow
	2: green
	3: no traffic light
	'''
	p = F.softmax(outputs[2], dim=1)
	traffic_light = np.argmax(p.detach().float().cpu().numpy().flatten())

	if (prob_blocked > 0.5 or traffic_light == 0 or traffic_light == 1):
		robot.left_motor.value = 0.0
		robot.right_motor.value = 0.0
	else:
		robot.left_motor.value = max(min(speed + steering, 1.0), 0.0)
		robot.right_motor.value = max(min(speed - steering, 1.0), 0.0)
	
	'''
	Show the result of free-blocked classification.
	'''
	if (prob_blocked < 0.5):
		cv2.putText(image, 'Free', (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	else:
		cv2.putText(image, 'Blocked', (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

	'''
	Show the result of traffic light color classification.
	'''
	if (traffic_light == 0):
		cv2.putText(image, 'Red', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
	elif (traffic_light == 1):
		cv2.putText(image, 'Yellow', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)	
	elif (traffic_light == 2):
		cv2.putText(image, 'Green', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

	draw_steering(image, angle)

	cv2.imshow('camera', image)
	key = cv2.waitKey(1)
	
	'''
	We can press the space (0x20) key to leave the loop.
	'''
	if (key == 0x20):
		break

camera.stop()
cv2.destroyAllWindows()