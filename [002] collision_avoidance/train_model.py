# Collision Avoidance - Train Model

'''
In this program, we'll train our image classifier to detect two 
classes `free` and `blocked`, which we'll use for avoiding collisions.
For this, we'll use a popular deep learning library PyTorch.
'''
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

'''
After collecting data in the folder of 'dataset', we use the 
`ImageFolder` dataset class available with the `torchvision.datasets` 
package. We attach transforms from the `torchvision.transforms`
package to prepare the data for training.
'''
dataset = datasets.ImageFolder(
	'dataset',
	transforms.Compose([
		transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
)

'''
Next, we split the dataset into training and test sets. The test set
will be used to verify the accuracy of the model we train.
'''
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 50, 50])

'''
We'll create two `DataLoader` instances, which provide utilities for
shuffling data, producing batches of images, and loading the samples
in parallel with multiple workers.
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
	shuffle=False,
	num_workers=0
)

'''
Now, we define the neural network we'll be training. The `torchvision`
package provides a collection of pre-trained models that we can use.

In a process called transfer learning, we can repurpose a pre-trained
model (trained on millions of images) for a new task that has possibly
much less data available

Important features that were learned in the original training of the
pre-trained model are usable for the new task. We'll use the `alexnet`
model.
'''
model = models.alexnet(pretrained=True)

'''
The `alexnet` model was originally trained for a dataset that had 1000
class labels, but our dataset only has two class labels! We'll replace
the final layer with a new, untrainedlayer that has two outputs.
'''
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)

'''
Finally, we transfer our model for execution. The default device is 
CPU. If you want to use GPU mode for training, please make sure that
your PyTorch version supports GPU. For GPU mode, the argument of
`torch.device` is 'cuda'.
'''
device = torch.device('cpu')
model = model.to(device)

'''
Using the code below, we'll train the neural network for 30 epochs, 
saving the best performing model after each epoch. Note that an epoch
is a full run through our data.
'''
NUM_EPOCHS = 30
BEST_MODEL_PATH = 'best_model.pth'
best_accuracy = 0.0

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(NUM_EPOCHS):

	for images, labels in iter(train_loader):
		images = images.to(device)
		labels = labels.to(device)
		optimizer.zero_grad()
		outputs = model(images)
		loss = F.cross_entropy(outputs, labels)
		loss.backward()
		optimizer.step()
	
	test_error_count = 0.0
	for images, labels in iter(test_loader):
		images = images.to(device)
		labels = labels.to(device)
		outputs = model(images)
		test_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))
	
	test_accuracy = 1.0 - float(test_error_count) / float(len(test_dataset))
	print('%d: %f' % (epoch, test_accuracy))
	if test_accuracy > best_accuracy:
		torch.save(model.state_dict(), BEST_MODEL_PATH)
		best_accuracy = test_accuracy

'''
Once that is finished, you should see a file `best_model.pth` in the
current working directory.
'''