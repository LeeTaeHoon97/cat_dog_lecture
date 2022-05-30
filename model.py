from graphviz import view
from numpy import short
import torch
import torch.nn.functional as F
import torch.nn as nn

class Residual_CNN(nn.Module):
	def __init__(self, in_channels=3, num_classes=2, _stride=2):
		super().__init__()
		self.conv1=nn.Sequential(
			nn.Conv2d(in_channels,out_channels=64,kernel_size= 7,stride=_stride,padding=3),
			nn.MaxPool2d(kernel_size=3,stride=_stride,padding=1)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(64,64,3,padding=1),
			nn.ReLU(),
			nn.Conv2d(64,64,3,padding=1)
		)
		self.conv3_1 = nn.Sequential(
			nn.Conv2d(64,128,3,padding=1),
			nn.ReLU(),
			nn.Conv2d(128,128,3,padding=1)
		)
		self.conv3_2 = nn.Sequential(
			nn.Conv2d(128,128,3,padding=1),
			nn.ReLU(),
			nn.Conv2d(128,128,3,padding=1)
		)
		self.conv4_1 = nn.Sequential(
			nn.Conv2d(128,256,3,padding=1),
			nn.ReLU(),
			nn.Conv2d(256,256,3,padding=1)
		)
		self.conv4_2 = nn.Sequential(
			nn.Conv2d(256,256,3,padding=1),
			nn.ReLU(),
			nn.Conv2d(256,256,3,padding=1)
		)
		self.conv5_1 = nn.Sequential(
			nn.Conv2d(256,512,3,padding=1),
			nn.ReLU(),
			nn.Conv2d(512,512,3,padding=1)
		)
		self.conv5_2 = nn.Sequential(
			nn.Conv2d(512,512,3,padding=1),
			nn.ReLU(),
			nn.Conv2d(512,512,3,padding=1)
		)
		self.shortcut1 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(128)
            )
		self.shortcut2 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(256)
            )
		self.shortcut3 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(512)
            )
		
		self.avg_pool= nn.AdaptiveAvgPool2d((1,1))				#kernnel size == feature map  => global avg pool

		self.fc1=nn.Sequential(
			nn.Linear(512,256),
			nn.ReLU()
		)
		self.fc2=nn.Sequential(
			nn.Linear(256,num_classes),
			nn.Sigmoid()							#pytroch의 cross entropy loss는 log_softmax + NLL_loss이므로 주석처리 				
		)

	


		self.relu=nn.ReLU()
	def residual_block(self,current_layer,shortcut):
		if shortcut.shape[1] != current_layer.shape[1]:
			shortcut
		return current_layer+shortcut


	def forward(self,x):
		
		x=self.conv1(x)
		shortcut=x

		x=self.conv2(x)
		x=self.residual_block(x,shortcut)
		x=self.relu(x)

		shortcut=x
		x=self.conv2(x)
		
		x=self.residual_block(x,shortcut)
		x=self.relu(x)
		shortcut=self.shortcut1(x)

	
		x=self.conv3_1(x)
		x=self.residual_block(x,shortcut)
		x=self.relu(x)
		shortcut=x
		x=self.conv3_2(x)
		x=self.residual_block(x,shortcut)
		x=self.relu(x)

		shortcut=self.shortcut2(x)

		x=self.conv4_1(x)
		x=self.residual_block(x,shortcut)
		x=self.relu(x)
		shortcut=x
		x=self.conv4_2(x)
		x=self.residual_block(x,shortcut)
		x=self.relu(x)

		shortcut=self.shortcut3(x)

		x=self.conv5_1(x)
		x=self.residual_block(x,shortcut)
		x=self.relu(x)
		shortcut=x
		x=self.conv5_2(x)
		x=self.residual_block(x,shortcut)
		
		x=self.relu(x)
		
		
		x=self.avg_pool(x)
		x=x.view(x.size(0),-1)
		x=self.fc1(x)
		x=self.fc2(x)

		return x
