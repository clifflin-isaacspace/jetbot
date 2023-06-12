# Multi-task Learning - Data Collection

'''
We assume that you've run through the samples of collision avoidance
and road following. Here, we want to combine the three basic features 
(collision avoidance, road following, and traffic light detection) 
together to have a fundamental self-driving car. For applications of
edge computing, the hardware resource is limited, so you may not be
able to run a large model or run multiple models. If the capacity of 
a model is large enough, one approach to address this issue can be 
multi-task learning. For our application, only one camera is used, so
the feature extracted from the image by a backbone model should be 
applicable for different tasks.

First, we need to modify the program of data collection in road following
to label images for multi-task learning. There are 3 tasks:
1. X-Y regression for road following.
2. Free-Blocked classification for collision avoidance.
3. Traffic light color classification for traffic light detection.
'''

'''
We import the same packages.
'''
from jetbot import Robot, Camera
import cv2
import os
import keyboard
import time
from uuid import uuid1

'''
Similarly, we define a mouse class and a callback function for handling
the mouse event.
'''
class Mouse(object):
	def __init__(self):
		self.x = 0
		self.y = 0
		self.clicked = False

def onMouse(event, x, y, flags, param):
	param.clicked = (event == 1)
	param.x = x
	param.y = y

'''
We use dictionary to define the task IDs and keys for labeling classes
'''

tasks = {
	'xy': 0,
	'free-blocked': 1,
	'traffic-light': 2
}

keys = {
	'free': 0,
	'blocked': 1,
	'red': 2,
	'yellow': 3,
	'green': 4,
}


'''
We create a folder to store the data and initialize necessities.
'''
DATASET_DIR = 'dataset_xy'

try:
	os.makedirs(DATASET_DIR)
except FileExistsError:
	print('Directories not created because they already exist')

camera = Camera.instance(width=224, height=224)
robot = Robot()
speed = 0.2

mouse = Mouse()
cv2.namedWindow('camera')
cv2.setMouseCallback('camera', onMouse, mouse)

while True:

	image = camera.value.copy()

	pt1 = (camera.width // 2, camera.height)
	pt2 = (mouse.x, mouse.y)

	image = cv2.circle(image, pt1, 8, (0, 0, 255), 4)
	image = cv2.circle(image, pt2, 8, (0, 255, 0), 4)
	image = cv2.line(image, pt1, pt2, (255, 0, 0), 4)
	
	task = -1
	label = -1
	x = 0.0
	y = 0.0

	'''
	If the mouse is clicked, record current mouse position, and assign
	the task ID.
	'''
	if (mouse.clicked):
		mouse.clicked = 0
		task = tasks['xy']
		x = mouse.x
		y = mouse.y

	'''
	We use keyboard to label the classification tasks.
	'''
	if   (keyboard.is_pressed(' ')):
		break
	elif   (keyboard.is_pressed(str(keys['free']))):
		label = keys['free']
	elif   (keyboard.is_pressed(str(keys['blocked']))):
		label = keys['blocked']
	elif   (keyboard.is_pressed(str(keys['red']))):
		label = keys['red']
	elif   (keyboard.is_pressed(str(keys['yellow']))):
		label = keys['yellow']
	elif   (keyboard.is_pressed(str(keys['green']))):
		label = keys['green']
	elif (keyboard.is_pressed('w')):
		robot.forward(speed)
	elif (keyboard.is_pressed('s')):
		robot.backward(speed)
	elif (keyboard.is_pressed('a')):
		robot.left(speed)
	elif (keyboard.is_pressed('d')):
		robot.right(speed)
	else:
		robot.stop()
	
	'''
	If use has labeled an image, save the image with a file name 
	containing label information.
	'''
	if ((task >= 0) or (label >= 0)):
		
		'''
		If task is -1, this image is for classification tasks.
		'''
		if (task < 0):
			'''
			If the label is 0 or 1, it is free-blocked classification task.
			
			If the label is 2, 3, or 4, it is traffic light color classification 
			task; then, the class ID should be corrected so that it ranges from 0
			to 2 
			'''
			if (label < 2):
				task = tasks['free-blocked']
			else:
				task = tasks['traffic-light']
				label -= 2
		'''
		Store the x-y coordinate, task and label in the file name.
		'''
		name = 'xy_%03d_%03d_%03d_%03d_%s.jpg' % (x, y, task, label, uuid1())
		print('save:', name)
		cv2.imwrite(DATASET_DIR + '/' + name, camera.value)
		
		'''
		When a new image is recored, check the image number of each task and class
		'''
		count = [0] * (len(keys) + 1)
		for filename in os.listdir(DATASET_DIR):
			t = int(filename.split("_")[3])
			c = int(filename.split("_")[4])
			if (t == 2):
				c += 2
			count[c] += 1
		for key in keys:
			print(key, count[keys[key]])
		print('xy', count[-1])
			
		time.sleep(0.2)
			
	cv2.imshow('camera', image)
	cv2.waitKey(1)

camera.stop()
cv2.destroyAllWindows()
