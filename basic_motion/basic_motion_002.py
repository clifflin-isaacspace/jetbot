# Controlling Motors Individually

''' 
Importing time package
'''
import time

'''
Importing the `Robot` class for controlling JetBot
'''
from jetbot import Robot

'''
Initializing a class instance of `Robot`
'''
robot = Robot()

'''
In the preivous sample, we saw how we can control the robot using 
commands like `left`, `right`, etc. But what if we want to set 
each motor speed individually? Well, there are two ways you can 
do this.
'''

'''
The first way is to call the `set_motors` method. For example, to 
turn along a left arch for a second, we couldset the left motor to
30% and the right motor to 60% like follows.
'''
print('Left arch 1')
robot.set_motors(0.3, 0.6)
time.sleep(1.0)
robot.stop()
'''
Great! You should see the robot move along a left arch. But actually,
there's another way that we could accomplish the same thing.
'''

'''
The `Robot` class has two attributes named `left_motor` and `right_motor`
that represent each motor individually. These attributes are `Motor` 
class instances, each which contains a `value` attribute. This `value` 
attribute is a `traitlet` which generates events when assigned a new 
value. In the motor class, we attach a function that upates the motor 
commands whenever the value changes.

So, to accomplish the exact same thing we did above, we could execute the
following.
'''
print('Left arch 2')
robot.left_motor.value = 0.3
robot.right_motor.value = 0.6
time.sleep(1.0)
robot.left_motor.value = 0.0
robot.right_motor.value = 0.0
