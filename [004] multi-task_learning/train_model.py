# Multi-task Learning - Train Model

'''
In this script, we'll train a neural network with 3 branches of outputs.
First, it is a set of x, y values corresponding to a target. Second, it is a 
free-blocked classifier for detecting obstacles in front of the robot. Third, 
it is a traffic light color classification for identifying traffic lights.
'''

'''
Similarly, we'll use PyTorch deep learning framework to train ResNet-18
neural network architecture model for road following, collision avoidance, 
and traffic light detection applications.
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

We create a dataset similar to that of road following. The difference is
that there are extra digits stored in the file name. One stands for the task
while the other stands for the class for classification tasks We need to 
parse them to train the free-blocked classifier and traffic light color 
classifier.
'''
def get_x(path, width):
	"""Gets the x value from the image filename"""
	return (float(int(path.split("_")[1])) - width/2) / (width/2)

def get_y(path, height):
	"""Gets the y value from the image filename"""
	return (float(int(path.split("_")[2])) - height/2) / (height/2)
	
def get_task(path):
	"""Gets the task from the image filename"""
	return int(path.split("_")[3])

def get_label(path):
	"""Gets the label from the image filename"""
	return int(path.split("_")[4])

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
		x = get_x(os.path.basename(image_path), width)
		y = get_y(os.path.basename(image_path), height)
		t = get_task(os.path.basename(image_path))
		c = get_label(os.path.basename(image_path))
	  
		if (float(np.random.rand(1)) > 0.5) and self.random_hflips:
			image = transforms.functional.hflip(image)
			x = -x
		
		image = self.color_jitter(image)
		image = transforms.functional.resize(image, (224, 224))
		image = transforms.functional.to_tensor(image)
		image = image.numpy()[::-1].copy()
		image = torch.from_numpy(image)
		image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		
		return image, torch.tensor([x, y]).float(), t, c
	
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
64 since we need larger batch size to have enough data for learning 
each task in a mini-batch. Batch size will be based on memory available 
with your GPU/RAM and it can impact accuracy of the model. Note that 
this configuration cannot run on Jetson Nano due to memory limitation, 
so it is recommended to use a personal computer to execute this training 
script.
'''
train_loader = torch.utils.data.DataLoader(
	train_dataset,
	batch_size=64,
	shuffle=True,
	num_workers=0
)

test_loader = torch.utils.data.DataLoader(
	test_dataset,
	batch_size=64,
	shuffle=False,
	num_workers=0
)

'''
Define Neural Network Model

We use ResNet-18 model available on PyTorch TorchVision.
'''
model = models.resnet18(pretrained=True)

'''
ResNet model has fully connect (fc) final layer with 512 as `in_features`
and we'll be training for x-y regression, free-blocked classification, and
traffic light color classification. We need 2 output channels for both
x-y regression and free-blocked classification. For traffic light color
classification, we need 4 channels. The first 3 stand for red, yellow, and 
green lights respectively while the last one stands for other situation.
We'll use all images labeled in the task of free-blocked classification to be
the class of the last channel. Therefore, `out_features` is set to 8.
'''
model.fc = torch.nn.Linear(512, 8)
device = torch.device('cpu')
model = model.to(device)

'''
Train Multi-task Learning

We train for 50 epochs and save best model if the loss is reduced and the 
accuracy is increased.
'''
NUM_EPOCHS = 50
BEST_MODEL_PATH = 'best_multi_task_model_xy.pth'
best_score = -1e9

optimizer = optim.Adam(model.parameters())

for epoch in range(NUM_EPOCHS):
	
	model.train()
	train_loss = 0.0
	for images, xylabels, t, c in iter(train_loader):

		images = images.to(device)
		xylabels = xylabels.to(device)
		c = c.to(device)

		optimizer.zero_grad()

		'''
		We split the output into 3 partitions. The first two outputs are used 
		for x-y regression. The next two outputs are used for free-bocked 
		classification. The last 4 outputs are used for traffic light color 
		classification.
		'''
		outputs = torch.split(model(images), [ 2, 2, 4 ], dim=1)

		'''
		We find the data with x-y regression task labeled, and build a mask to
		weight the mean square error loss.
		'''

		(i, ) = np.where(t == 0)
		num = len(i)

		if (num > 0):
			w = np.zeros(len(t))
			w[i] = 1
			w = torch.FloatTensor(w).to(device)
			loss1 = torch.sum(w * torch.sum((outputs[0] - xylabels)**2, dim=1)) / num
		else:
			loss1 = 0.0

		'''
		Similarly, we find the data with free-blocked classification task labeled,
		and use these data to compute the cross entropy loss.
		'''
		(i, ) = np.where(t == 1)
		num = len(i)

		if (num > 0):
			loss2 = F.cross_entropy(outputs[1][i], c[i])
		else:
			loss2 = 0.0

		'''
		We find the data with traffic light color classification task labeled, and
		change the labels of data in free-blocked classification task; then, we use
		all of these data to compute the multi-class cross entropy loss.
		'''
		(j, ) = np.where(t == 2)
		num = len(j)

		if (num > 0):
			c[i] = 3
			ind = np.concatenate([i, j])
			loss3 = F.cross_entropy(outputs[2][ind], c[ind])
		else:
			loss3 = 0.0

		'''
		All losses above are summed to be a total loss, and we use it to train our
		model.
		'''
		loss = loss1 + loss2 + loss3
		train_loss += float(loss)
		loss.backward()
		optimizer.step()

	train_loss /= len(train_loader)
	
	'''
	In the section of testing dataset, we use the same flow to compute the testing 
	loss of x-y regression, and compute the testing accuracies of both free-blocked 
	classification and traffic light color classification.
	'''
	model.eval()
	test_loss = 0.0
	test_correct1_count = 0
	test_correct2_count = 0
	count1 = 0
	count2 = 0
	for images, xylabels, t, c in iter(test_loader):

		images = images.to(device)
		xylabels = xylabels.to(device)
		c = c.to(device)

		outputs = torch.split(model(images), [ 2, 2, 4 ], dim=1)

		(i, ) = np.where(t == 0)
		num = len(i)

		if (num > 0):
			w = np.zeros(len(t))
			w[i] = 1
			w = torch.FloatTensor(w).to(device)
			loss1 = torch.sum(w * torch.sum((outputs[0] - xylabels)**2, dim=1)) / num
		else:
			loss1 = 0.0
			
		test_loss += float(loss1)

		(i, ) = np.where(t == 1)
		num = len(i)

		if (num > 0):
			p = outputs[1][i].argmax(dim=1)
			test_correct1_count += torch.sum(torch.eq(p, c[i]))
			count1 += num

		(j, ) = np.where(t == 2)
		c[i] = 3
		ind = np.concatenate([i, j])
		p = outputs[2][ind].argmax(dim=1)
		test_correct2_count += torch.sum(torch.eq(p, c[ind]))
		count2 += num + len(j)

	test_loss /= len(test_loader)
	test_accuracy1 = float(test_correct1_count) / float(count1)
	test_accuracy2 = float(test_correct2_count) / float(count2)
	
	'''
	We simply use the average of the testing accuracies and the inverted testing loss
	to be the evaluation score (test_score). The higher the score is, the better
	the model should be. Also, you can define your own evaluation score to choose
	the best model in the training. 
	'''
	test_score = (test_accuracy1 + test_accuracy2 + (1.0 - test_loss)) / 3

	print('%f, %f, %f, %f, %f' % (train_loss, test_loss, test_accuracy1, test_accuracy2, test_score))

	if test_score > best_score:
		torch.save(model.state_dict(), BEST_MODEL_PATH)
		best_score = test_score

'''
Once the model is trained, it will generate `best_multi_task_model_xy.pth`
file which you can use for inferencing in the live demo python script.
'''