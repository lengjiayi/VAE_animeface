import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mping
import csv
import random
from PIL import Image


def getrecon():
	sample = []
	text = open('random.txt','r',encoding='utf-8')
	row = csv.reader(text, delimiter=',')
	for r in row:
		pict = []
		for pix in r:
			if pix:
				pict.append(int(pix))
		pict = np.array(pict)
		sample.append([pict.reshape(30,30)])
	sample = np.array(sample)
	return sample



class StackNN(nn.Module):
	def __init__(self):
		super(StackNN, self).__init__()
		self.remap = nn.Sequential(
			nn.ConvTranspose2d(1,10,5,stride=2,padding=0),
			nn.ReLU(),
			nn.Conv2d(10,5,5),#3*59*59
			nn.ReLU(),
			)
		self.fc = nn.Sequential(
			nn.Linear(5*59*59,5*30*30),
			nn.Linear(5*30*30,3*30*30),
			)
	def forward(self, x):
		eps = torch.randn(x.size(0),x.size(1),x.size(2),x.size(3))
		x=x+eps
		x=self.remap(x)
		x=x.view(x.shape[0],5*59*59)
		x=self.fc(x)
		x=x.view(x.shape[0],3,30,30)
		return x




def getImg(array):
	array=array.reshape(3,30*30)
	img=[]
	for i in range(30*30):
		img.append([array[0][i],array[1][i],array[2][i]])
	img=np.array(img)
	img=np.clip(img,0,1)
	return img.reshape(30,30,3)


def coloring():
	colorful = stacknn(torch.from_numpy(recon).float())
	i=0
	for imgtensor in colorful:
		img = imgtensor.detach().numpy()
		img = getImg(img)
		img = (img*255).clip(0,255)
		img = Image.fromarray(np.uint8(img)).resize((100,100))
		img.save('./RandomColorful/'+str(i)+'.png')
		i=i+1






recon = getrecon() #(1044, 1, 30, 30)

stacknn=StackNN()
stacknn.load_state_dict(torch.load('stacknn.pth'))

coloring()