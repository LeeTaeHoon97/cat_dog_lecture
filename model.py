import torch
import torch.nn.functional as F
import torch.nn as nn

class Residual_CNN(nn.Module):
	def __init__(self, in_channels=3, num_classes=2, _stride=2):
		super().__init__()
		self.conv1=nn.Sequential(
			nn.Conv2d(in_channels,out_channels=64,kernel_size= 7,stride=_stride),
			nn.MaxPool2d(kernel_size=3,stride=_stride)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(64,64,3),
			nn.ReLU(),
			nn.Conv2d(64,64,3)
		)
		self.conv3_1 = nn.Sequential(
			nn.Conv2d(64,128,3),
			nn.ReLU(),
			nn.Conv2d(128,128,3)
		)
		self.conv3_2 = nn.Sequential(
			nn.Conv2d(128,128,3),
			nn.ReLU(),
			nn.Conv2d(128,128,3)
		)
		self.conv4_1 = nn.Sequential(
			nn.Conv2d(128,256,3),
			nn.ReLU(),
			nn.Conv2d(256,256,3)
		)
		self.conv4_2 = nn.Sequential(
			nn.Conv2d(256,256,3),
			nn.ReLU(),
			nn.Conv2d(256,256,3)
		)
		self.conv5_1 = nn.Sequential(
			nn.Conv2d(256,512,3),
			nn.ReLU(),
			nn.Conv2d(512,512,3)
		)
		self.conv5_2 = nn.Sequential(
			nn.Conv2d(512,512,3),
			nn.ReLU(),
			nn.Conv2d(512,512,3)
		)
		
		self.avg_pool= nn.AvgPool2d(1)				#kernnel size == feature map  => global avg pool

		self.fc1=nn.Sequential(
			nn.Linear(512,1000),
			nn.ReLU()
		)
		self.fc2=nn.Sequential(
			nn.Linear(1000,num_classes)
			# nn.Softmax()							#pytroch의 cross entropy loss는 log_softmax + NLL_loss이므로 주석처리 				
		)
	def residual_block(current_layer,shortcut):
		return nn.ReLU(current_layer+shortcut)
		

	def forward(self,x):
		x=self.conv1(x)

		shortcut=x

		x=self.conv2(x)
		x=self.residual_block(x,shortcut)
		shortcut=x
		x=self.conv2(x)
		x=self.residual_block(x,shortcut)

		shortcut=x

		x=self.conv3_1(x)
		x=self.residual_block(x,shortcut)
		shortcut=x
		x=self.conv3_2(x)
		x=self.residual_block(x,shortcut)

		shortcut=x

		x=self.conv4_1(x)
		x=self.residual_block(x,shortcut)
		shortcut=x
		x=self.conv4_2(x)
		x=self.residual_block(x,shortcut)

		shortcut=x

		x=self.conv5_1(x)
		x=self.residual_block(x,shortcut)
		shortcut=x
		x=self.conv5_2(x)
		x=self.residual_block(x,shortcut)

		x=self.avg_pool(x)
		x=self.fc1(x)
		x=self.fc2(x)

		return x
