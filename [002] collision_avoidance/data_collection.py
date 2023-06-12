# Collision Avoidance - Data Collection

'''
If you can run through the basic motion programs, hopefully you're
enjoying how easy it can be to make your JetBot move around! That's
very cool! But what's even cooler, is making JetBot move around by
itself!

This is a super hard task. That has many different approaches, but
the whole problem is usually broken down into easier sub-problems.
It could be argued that one of the most important sub-problems to 
solve, is the problem of preventing the robot entering dangerous
situations! We're calling this "collision avoidance".

In this set of programs, we're going to attempt to solve the problem
using deep learning and a single, very versatile, sensor: the camera.
You'll see how with a neural network, camera, and NVIDIA Jetson Nano,
we can teach the robot a very useful behavior!

The approach we take to avoiding collisions is to create a virtual
"safety bubble" around the robot. Within this safety bubble, the 
robot is able to spin in a circle without hitting any objects (or
other dangerous situations like falling off a ledge).

Of course, the robot is limited by what's in it's field of vision,
and we can't prevent objects from being placed behind the robot, etc.
But, we can prevent the robot from entering these scenarios itself.

The way we'll do this is super simple:

First, we'll manually place the robot in scenarios where it's 
"safety bubble" is violated, and label these scenarios `blocked`.
We save a snapshot of what the robot sees along with this label.

Second, we'll manually place the robot in scenarios where it's safe
to move forward a bit, and label these scenarios `free`. Likewise, 
we save a snapshot along with this label.

That's all that we'll do in this program: data collection. Once we
have lots of images and labels, we'll use this data to train a neural
network to predict whether the robot's safety bubble is being violated
based off of the image it sees. We'll use this to implement a simple 
collision avoidance behavior in the end.
'''

''' 
Importing time package
'''
import time

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
For convenient, we can define a function to handle the progress of
saving an image. 
'''
def save_snapshot(directory, image):
	image_path = os.path.join(directory, str(uuid1()) + '.jpg')
	cv2.imwrite(image_path, image)

'''
First, let's create a few directories where we'll store all our 
data. We'll create a folder `dataset` that will contain two sub-folders
`free` and `blocked`, where we'll place the images for each scenario.
'''
blocked_dir = 'dataset/blocked'
free_dir    = 'dataset/free'

try:
	os.makedirs(free_dir)
	os.makedirs(blocked_dir)
except FileExistsError:
	print('Directories not created because they already exist')

'''
Let's initialize our camera. Since our neural network takes 
a 224x224 pixel image as input, we'll set our camera to that size 
to minimize the file size of our dataset (we've tested that it works
for this task). In some scenarios, it may be better to collect data 
in a larger image size and downscale to the desired size later.
'''
camera = Camera.instance(width=224, height=224)

'''
Initializing a class instance of `Robot`
'''
robot = Robot()

'''
We can declare a variable named `speed` to set the base speed 
of the robot.
'''
speed = 0.2

while True:

	'''
	We can get the image frame from the camera and display it 
	through the OpenCV widget. Given a name of window and an image, 
	the function `cv2.imshow` can show the image on the window. 
	In the following, the image will be drawn in a window named 
	'camera'. The function `cv2.waitKey` block the code for certain
	period to wait a key event and also swap the buffer to show 
	the image. For example, `cv2.waitKey(1)` will block the code 
	for 1 millisecond and show the image on the window.
	'''
	image = camera.value
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
	elif (keyboard.is_pressed('f')):
		save_snapshot(free_dir, image)
		print('free_count', len(os.listdir(free_dir)))
		time.sleep(0.1)
	elif (keyboard.is_pressed('b')):
		save_snapshot(blocked_dir, image)
		print('blocked_count', len(os.listdir(blocked_dir)))
		time.sleep(0.1)
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
