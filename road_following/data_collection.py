# Road Following - Data Collection

'''
If you've run through the collision avoidance sample, you should
be familiar following three steps:
	1. Data collection
	2. Training
	3. Deyployment
Here, we'll fo the same exact thing! Except, instead of classification,
you'll learn a different fundamental technique, 'regression', that
we'll use to enable JetBot to follow a road (or really, any path or 
target point).
	1. Place the JetBot in different positions on a path (offset from 
	   center, different angles, etc). Remember from collision avoidance,
	   data variation is key!
	2. Display the live camera feed from the robot.
	3. Using the mouse to place a 'green dot', which corresponds to the
	   target direction we want the robot to travel on the image.
	4. Store the X, Y values of this 'green dot' along with the image 
	   from the robot's camera.
Then, in the training script, we'll train a neural network to predict 
the X, Y values of our label. In the live demo, we'll use the predicted
X, Y values to compute an approximate steering value (it's not 'exactly'
an angle, as that would require image calibration, but it's roughly 
proportional to the angle, so our controller will work fine).

So, how do you decide exactly where to place the target for this example?
Here is a guide we think help
	1. Look at the lvie video feed from the camera
	2. Imagine the path that the robot should follow (try to approximate
	   the distance it needs to avoid running off road etc.)
	3. Place the target as far along this path as it can go so that the
	   robot could head straight to the target without 'running off' the 
	   road. For example, if we're on a very straight road, we could place
	   it at the horizon. If we're on a sharp turn, it may need to be placed
	   closer to the robot, so it doesn't run out of boundaries.
Assuming our deep learning model works as intended, these labeling guidelines 
should ensure the following:
	1. The robot can safely travel directly towards the target (without going
	   out of bounds etc.)
	2. The target will continously progress along our imagined path.
What we get is a 'carrot on a stick' that moves along our desired trajectory.
Deep learning decides where to place the carrot, and JetBot just follows it.
'''

'''
Now, we have to import `Robot` class and `Camera` class to control 
the robot and get the image frames from the camera.
'''
from jetbot import Robot, Camera

'''
To display and save an image, we use OpenCV package named `cv2` to 
do that.
'''
import cv2

'''
Since we have to build folders for saving data, we can use `os` 
package to do that.
'''
import os

'''
Keyboard is the most convenient tool for controlling robot in a 
simulator. We use the `keyboard` package to handle the key event.
'''
import keyboard

'''
To make sure we don't repeat any file names (even across different 
machines!), we'll use `uuid` package in python, which defines the 
`uuid1` method to generate a unique identifier. This unique identifier 
is generated from information like the current time and the machine 
address.
'''
from uuid import uuid1


'''
For convenient, let's define an object to store the status of the 
mouse.
'''
class Mouse(object):
	def __init__(self):
		self.x = 0
		self.y = 0
		self.clicked = False

'''
Next, we define a callback function to update the state of the mouse.
'''
def onMouse(event, x, y, flags, param):
	param.clicked = (event == 1)
	param.x = x
	param.y = y

'''
Similarly, let's create a directory to store data.
'''
DATASET_DIR = 'dataset_xy'

try:
	os.makedirs(DATASET_DIR)
except FileExistsError:
	print('Directories not created because they already exist')

'''
Initializing the camera and robot.
'''
camera = Camera.instance(width=224, height=224)
robot = Robot()

'''
We can declare a variable named `speed` to set the base speed 
of the robot.
'''
speed = 0.2

'''
Declare a mouse object and setup the callback function. We have
to bind the callback to a named window which will be used for
displaying the preview of camera.
'''
mouse = Mouse()
cv2.namedWindow('camera')
cv2.setMouseCallback('camera', onMouse, mouse)

while True:

	'''
	We need to have a copy of camera frame since we'll draw widgets
	on the image and do not want to change the original one.
	'''
	image = camera.value.copy()

	'''
	pt1 is the point at the center-bottom of the image. pt2 is the 
	mouse position.
	'''
	pt1 = (camera.width // 2, camera.height)
	pt2 = (mouse.x, mouse.y)

	'''
	Draw pt1, pt2, and their connection.
	'''
	image = cv2.circle(image, pt1, 8, (0, 0, 255), 4)
	image = cv2.circle(image, pt2, 8, (0, 255, 0), 4)
	image = cv2.line(image, pt1, pt2, (255, 0, 0), 4)

	'''
	If the mouse is clicked, record current mouse position and save
	the image. The mouse position will be in the image name.
	'''
	if (mouse.clicked):
		mouse.clicked = False
		name = 'xy_%03d_%03d_%s.jpg' % (mouse.x, mouse.y, uuid1())
		print('save:', name)
		cv2.imwrite(DATASET_DIR + '/' + name, camera.value)

	cv2.imshow('camera', image)
	cv2.waitKey(1)

	'''
	Like basic_motion_003.py, we use keyboard to control the robot.
	Besides, we can also use keyboard to label the image and save it.
	We use key 'f' for the label of free and key 'b' for the label 
	of blocked.
	'''
	if   (keyboard.is_pressed(' ')):
		break
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

camera.stop()
cv2.destroyAllWindows()
