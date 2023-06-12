# Road Following - Live Demo with Traffic Light Detection

'''
If you've finish data_collection.py, train_model.py, and live_demo.py, 
you should be familiar with the basic concept of road following. For the
application of self-driving car, it should follow some rules to avoid 
traffic accident. One essential feature is detecting the traffic light.
The car must stop when detecting yellow or red light until the light turns 
green. This program is the extension of live_demo.py. To make it simpler, 
we implement a function for detecting traffic light based on basic image 
processing technique. In the main loop of this program, JetBot detects the
traffic light to determine whether it can move forward or not, which 
demonstrates the basic idea of self-driving.
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

We'll assume that you've already had `best_steering_model_xy.pth`. Now, 
you should initialize the PyTorch model and load the trained wights from 
`best_steering_model_xy.pth`.
'''

model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)

model.load_state_dict(torch.load('best_steering_model_xy.pth'))

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
Implement a function for detecting traffic light

The core technique of traffic light detection in this program is basic
image processing. We use color detection to identify the backbone of the
traffic light, and the light colors. First, we have to define the color 
range of each item. For detecting backbone, we set the minimum width and 
height to filter out some small spots detected by color detection. 
If a backbone is detected, we will detect the light colors in the area of 
backbone. Next, we can compute the pixel number of each light color.
If the pixel number of a light color is more than a threshold, that light
color should be the current light status. Then, the function will return
the result and the bounding box of the backbone.
'''

# Traffic light backbone color range
TRAFFIC_LIGHT_COLOR_MIN = ( 0,  0,  0)
TRAFFIC_LIGHT_COLOR_MAX = (50, 50, 50)

# Minimum size of traffic light backbone
TRAFFIC_LIGHT_MIN_WIDTH = 20
TRAFFIC_LIGHT_MIN_HEIGHT = 80

# Red light color range
TRAFFIC_LIGHT_RED_MIN = (  0,   0, 200)
TRAFFIC_LIGHT_RED_MAX = (100, 100, 255)

# Yellow light color range
TRAFFIC_LIGHT_YELLOW_MIN = (  0, 128, 128)
TRAFFIC_LIGHT_YELLOW_MAX = (100, 255, 255)

# Green light color range
TRAFFIC_LIGHT_GREEN_MIN = (  0, 128,   0)
TRAFFIC_LIGHT_GREEN_MAX = (100, 255, 100)

# Threshold of pixel number
TRAFFIC_LIGHT_PIXEL_THRESH = 5000 # pixels

# The macro of return value
TRAFFIC_LIGHT_NONE = 0
TRAFFIC_LIGHT_RED = 1
TRAFFIC_LIGHT_YELLOW = 2
TRAFFIC_LIGHT_GREEN = 3

def detect_traffic_light(image):

	'''
	We initialize the results here.
	'''
	ret = TRAFFIC_LIGHT_NONE
	x = 0
	y = 0
	w = 0
	h = 0

	width = image.shape[1]
	height = image.shape[0]

	'''
	We assume that the traffic light is always on the right, so we crop the right
	half image for processing.
	'''
	roi_image = image[0:height, width//2:width, :]

	'''
	We detect the color of traffic light backbone and find all possible contours
	in the image. If some contours are detected, we can do further process to
	identify the light color.
	'''
	bin_map = cv2.inRange(roi_image, TRAFFIC_LIGHT_COLOR_MIN, TRAFFIC_LIGHT_COLOR_MAX)
	contours, hierarchy = cv2.findContours(bin_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

	if  (len(contours) > 0):

		'''
		We get the rectangle area of the maximum contour. If the size is large 
		enough, we detect the color of each light color in that rectangle.
		'''
		cnt = max(contours, key=cv2.contourArea)
		x, y, w, h = cv2.boundingRect(cnt)
		
		if (w > TRAFFIC_LIGHT_MIN_WIDTH and h > TRAFFIC_LIGHT_MIN_HEIGHT):
			
			color = roi_image[y:y+h, x:x+w, :]
			
			'''
			We sum the number of pixels of each light color, and check that if 
			any of which is larger than the threshold. Since yellow is the 
			combination of red and green color, it should be check first, or 
			the detection result will be either red or green.
			'''
			red = np.sum(cv2.inRange(color, TRAFFIC_LIGHT_RED_MIN, TRAFFIC_LIGHT_RED_MAX))
			yellow = np.sum(cv2.inRange(color, TRAFFIC_LIGHT_YELLOW_MIN, TRAFFIC_LIGHT_YELLOW_MAX))
			green = np.sum(cv2.inRange(color, TRAFFIC_LIGHT_GREEN_MIN, TRAFFIC_LIGHT_GREEN_MAX))
			
			if (yellow > TRAFFIC_LIGHT_PIXEL_THRESH):
				ret = TRAFFIC_LIGHT_YELLOW
			elif (red > TRAFFIC_LIGHT_PIXEL_THRESH):
				ret = TRAFFIC_LIGHT_RED
			elif (green > TRAFFIC_LIGHT_PIXEL_THRESH):
				ret = TRAFFIC_LIGHT_GREEN

	return ret, (x+width//2, y, w, h)

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
	4. Try to detect traffic light.
	5. Stop the robot if red or yellow light is detected, or control the motors 
	   using  proportional / derivative control. (PD control)
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

	'''
	We copy the image from camera since we want to draw some information
	on the image and do not want to change the original one.
	'''
	image = camera.value.copy()

	xy = model(preprocess(image)).detach().float().cpu().numpy().flatten()
	x = xy[0]
	y = (0.5 - xy[1]) / 2.0

	angle = np.arctan2(x, y)
	pid = angle * steering_gain + (angle - angle_last) * steering_dgain
	angle_last = angle

	steering = pid + steering_bias

	light, bbox = detect_traffic_light(image)
	
	'''
	If the traffic light is red or yellow, stop the robot.
	'''
	if (light == TRAFFIC_LIGHT_RED or light == TRAFFIC_LIGHT_YELLOW):
		robot.left_motor.value = 0.0
		robot.right_motor.value = 0.0
	else:
		robot.left_motor.value = max(min(speed + steering, 1.0), 0.0)
		robot.right_motor.value = max(min(speed - steering, 1.0), 0.0)

	'''
	We show the result of traffic light detection on the image.
	'''
	if (light == TRAFFIC_LIGHT_RED):
		color = (0, 0, 255)
		cv2.putText(image, 'Red', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
	elif (light == TRAFFIC_LIGHT_YELLOW):
		color = (0, 255, 255)
		cv2.putText(image, 'Yellow', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
	elif (light == TRAFFIC_LIGHT_GREEN):
		color = (0, 255, 0)
		cv2.putText(image, 'Green', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

	if (light != TRAFFIC_LIGHT_NONE):
		cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color, 2)

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