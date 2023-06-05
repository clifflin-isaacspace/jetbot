# Road Following - Train Model

'''
In this script, we'll train a neural network to take an input
image, and output a set of x, y values corresponding to a target.
'''

'''
We'll be using PyTorch deep learning framework to train ResNet-18
neural network architecture model for road following application.
'''
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import glob
import PIL.Image
import os
import numpy as np


'''
Crete Dataset Instance

Here, we create a custom `torch.utils.data.Dataset` implementation,
which implements the `__len__` and `__getitem__` functions. This 
class is responsible for loading images and parsing the x, y values 
from the image filenames. Because we implement the `torch.utils.data.Dataset`
class, we can use all of the torch data utilities.

We hard coded some transformations (like color jitter) into our dataset.
We made random horizontal flips optional (in case you want to follow
a non-symmetric path, like a road where we need to `stay right`). If
it doesn't matter whether your robot follows some convention, you
could enable flips to augment the dataset.
'''
def get_x(path, width):
	"""Gets the x value from the image filename"""
	return (float(int(path.split("_")[1])) - width/2) / (width/2)

def get_y(path, height):
	"""Gets the y value from the image filename"""
	return (float(int(path.split("_")[2])) - height/2) / (height/2)

class XYDataset(torch.utils.data.Dataset):
	
	def __init__(self, directory, random_hflips=False):
		self.directory = directory
		self.random_hflips = random_hflips
		self.image_paths = glob.glob(os.path.join(self.directory, '*.jpg'))
		self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)
	
	def __len__(self):
		return len(self.image_paths)
	
	def __getitem__(self, idx):
		image_path = self.image_paths[idx]
		
		image = PIL.Image.open(image_path)
		width, height = image.size
		x = float(get_x(os.path.basename(image_path), width))
		y = float(get_y(os.path.basename(image_path), height))
	  
		if (float(np.random.rand(1)) > 0.5) and self.random_hflips:
			image = transforms.functional.hflip(image)
			x = -x
		
		image = self.color_jitter(image)
		image = transforms.functional.resize(image, (224, 224))
		image = transforms.functional.to_tensor(image)
		image = image.numpy()[::-1].copy()
		image = torch.from_numpy(image)
		image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		
		return image, torch.tensor([x, y]).float()
	
dataset = XYDataset('dataset_xy', random_hflips=False)

'''
Split Dataset into Train and Test Sets

Once we read datasets, we'll split dataset in train and test sets. In
this example, we split train and test a 90%-10%. The test set will be
used to verify the accuracy of the model we train.
'''
test_percent = 0.1
num_test = int(test_percent * len(dataset))
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - num_test, num_test])

'''
Create Data Loader to Load Data in Batches

We use `DataLoader` class to load data in batches, shuffle data and 
allow using multi-subprocess. In this example, we use batch size of 
8. Batch size will be based on memory available with your GPU/RAM
and it can impact accuracy of the model.
'''
train_loader = torch.utils.data.DataLoader(
	train_dataset,
	batch_size=8,
	shuffle=True,
	num_workers=0
)

test_loader = torch.utils.data.DataLoader(
	test_dataset,
	batch_size=8,
	shuffle=True,
	num_workers=0
)

'''
Define Neural Network Model

We use ResNet-18 model available on PyTorch TorchVision.

In a process called transfer learning, we can repurpose a pre-trained
model (trained on millions of images) for a new task that has possibly
much less data available.

More details on ResNet-18 : 
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

More Details on Transfer Learning: 
https://www.youtube.com/watch?v=yofjFQddwHE
'''
model = models.resnet18(pretrained=True)

'''
ReNet model has fully connect (fc) final layer with 512 as `in_features`
and we'll be training for x and y regression thus `out_features` as 2.

Finally, we transfer our model for execution on CPU. If you have installed
PyTorch with GPU support, it is better to execute on GPU.
'''
model.fc = torch.nn.Linear(512, 2)
device = torch.device('cpu') # input 'cuda' for using GPU
model = model.to(device)

'''
Train Regression

We train for 50 epochs and save best model if the loss is reduced.
'''
NUM_EPOCHS = 50
BEST_MODEL_PATH = 'best_steering_model_xy.pth'
best_loss = 1e9

optimizer = optim.Adam(model.parameters())

for epoch in range(NUM_EPOCHS):
	
	model.train()
	train_loss = 0.0
	for images, labels in iter(train_loader):
		images = images.to(device)
		labels = labels.to(device)
		optimizer.zero_grad()
		outputs = model(images)
		loss = F.mse_loss(outputs, labels)
		train_loss += float(loss)
		loss.backward()
		optimizer.step()
	train_loss /= len(train_loader)
	
	model.eval()
	test_loss = 0.0
	for images, labels in iter(test_loader):
		images = images.to(device)
		labels = labels.to(device)
		outputs = model(images)
		loss = F.mse_loss(outputs, labels)
		test_loss += float(loss)
	test_loss /= len(test_loader)
	
	print('%f, %f' % (train_loss, test_loss))
	if test_loss < best_loss:
		torch.save(model.state_dict(), BEST_MODEL_PATH)
		best_loss = test_loss

'''
Once the model is trained, it will generate `best_steering_model_xy.pth`
file which you can use for inferencing in the live demo python script.
'''