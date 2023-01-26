import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


# for MNIST 32*32
class CNN_Net(nn.Module):

	def __init__(self, device=None):
		super(CNN_Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 64, 3, 1)
		self.conv2 = nn.Conv2d(64, 16, 7, 1)
		self.fc1 = nn.Linear(4 * 4 * 16, 200)
		self.fc2 = nn.Linear(200, 10)

	def forward(self, x):
		x = x.view(-1, 1, 32, 32)
		x = torch.tanh(self.conv1(x))
		x = F.max_pool2d(x, 2, 2)
		x = torch.tanh(self.conv2(x))
		x = F.max_pool2d(x, 2, 2)
		x = x.view(-1, 4 * 4 * 16)
		x = torch.tanh(self.fc1(x))
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)


# for MNIST 32*32 LogReg
class MNIST_LogisticRegression(nn.Module):

	def __init__(self, input_dim=1024, output_dim=10, device=None):
		super(MNIST_LogisticRegression, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.linear = torch.nn.Linear(self.input_dim, self.output_dim)

	def forward(self, x):
		x = x.view(-1,  1024)
		outputs = self.linear(x)
		return F.log_softmax(outputs, dim=1)

# for MNIST 32*32
class MLP_Net(nn.Module):

	def __init__(self, device=None):
		super(MLP_Net, self).__init__()
		self.fc1 = nn.Linear(1024, 128)
		self.fc2 = nn.Linear(128, 64)
		self.fc3 = nn.Linear(64, 10)

	def forward(self, x):
		x = x.view(-1,  1024)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return F.log_softmax(x, dim=1)



# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# LeNet
class CNNCifar(nn.Module):
	def __init__(self, device=None):
		super(CNNCifar, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)
	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return F.log_softmax(x, dim=1)
 
# https://www.tensorflow.org/tutorials/images/cnn
class CNNCifar_TF(nn.Module):
	def __init__(self, device=None):
		super(CNNCifar_TF, self).__init__()
		self.conv1 = nn.Conv2d(3, 32, 3)
		self.conv2 = nn.Conv2d(32, 64, 3)
		self.conv3 = nn.Conv2d(64, 64, 3)
		# self.bn1 = nn.BatchNorm2d(32)
		# self.bn2 = nn.BatchNorm2d(64)
		# self.bn3 = nn.BatchNorm2d(64)
		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(64 * 4 * 4, 64)
		self.fc2 = nn.Linear(64, 10)
	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = F.relu(self.conv3(x))
		x = x.view(-1, 64 * 4 * 4)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)

from torchvision import models
class ResNet18_torch(nn.Module):
	def __init__(self, pretrained=False, device=None):
		super().__init__()
		self.resnet = models.resnet18(pretrained=pretrained)

		num_ftrs = self.resnet.fc.in_features
		self.resnet.fc = nn.Linear(num_ftrs, 10)  # make the change

		self.resnet.conv1 = torch.nn.Conv2d(
			3, 64, kernel_size=3, stride=1, padding=1, bias=False
		)
		self.resnet.maxpool = torch.nn.Identity()

	def forward(self, x):
		x = self.resnet(x)
		x = F.log_softmax(x, dim=1)
		return x



class CNN_Text(nn.Module):
	
	def __init__(self, args=None, device=None):
		super(CNN_Text,self).__init__()

		
		self.args = args
		self.device = device
		
		V = args['embed_num']
		D = args['embed_dim']
		C = args['class_num']
		Ci = 1
		Co = args['kernel_num']
		Ks = args['kernel_sizes']

		self.embed = nn.Embedding(V, D)
		self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
		'''
		self.conv13 = nn.Conv2d(Ci, Co, (3, D))
		self.conv14 = nn.Conv2d(Ci, Co, (4, D))
		self.conv15 = nn.Conv2d(Ci, Co, (5, D))
		'''
		self.dropout = nn.Dropout(0.5)
		# self.dropout = nn.Dropout(args.dropout)
		self.fc1 = nn.Linear(len(Ks)*Co, C)

	def conv_and_pool(self, x, conv):
		x = F.relu(conv(x)).squeeze(3) #(N,Co,W)
		x = F.max_pool1d(x, x.size(2)).squeeze(2)
		return x


	def forward(self, x):

		x = self.embed(x) # (W,N,D)
		# x = x.permute(1,0,2) # -> (N,W,D)
		# permute during loading the batches instead of in the forward function
		# in order to allow nn.DataParallel

		if not self.args or self.args['static']:
			x = Variable(x).to(self.device)

		x = x.unsqueeze(1) # (W,Ci,N,D)

		x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
		x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
		x = torch.cat(x, 1)
		'''
		x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
		x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
		x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
		x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
		'''
		x = self.dropout(x) # (N,len(Ks)*Co)
		logit = self.fc1(x) # (N,C)
		return F.log_softmax(logit, dim=1)
		# return logit