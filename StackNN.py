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


learning_rate = 0.0001
epoch_num = 20
batch_size = 80

criterion = nn.MSELoss()


def getrecon():
	sample = []
	text = open('reconstruct.txt','r',encoding='utf-8')
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


def getraw():
	sample = []
	for i in range(1,1045):
		img=mping.imread('./small/face'+str(i)+'.png')
		reform=[img[:,:,0],img[:,:,1],img[:,:,2]]
		sample.append(reform)
	return np.array(sample)


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


def vitest():
	for recon,raw in testset:
		outputset = stacknn(recon.float())
		i=0
		for output in outputset:
			i=i+1
			face = output.detach().numpy()
			face = getImg(face)
			plt.subplot(6,6,i+12)
			plt.xticks([])
			plt.yticks([])
			plt.imshow(face)
		i=0
		for rawpic in raw:
			i=i+1
			face = rawpic.detach().numpy()
			face = getImg(face)
			plt.subplot(6,6,i+24)
			plt.xticks([])
			plt.yticks([])
			plt.imshow(face)
		i=0
		for reconpic in recon:
			i=i+1
			face = reconpic.detach().numpy().reshape(30,30)
			face = face.reshape(30,30)
			plt.subplot(6,6,i)
			plt.xticks([])
			plt.yticks([])
			plt.imshow(face,cmap='gray')
		break
	mng = plt.get_current_fig_manager()
	mng.window.state('zoomed')
	plt.subplots_adjust(wspace=0,hspace=0,left=None,right=None,bottom=None,top=None)
	plt.show()


def loss_func(recon_x, x, target):
	colorloss = criterion(recon_x.float(), target.float())
	gray = (recon_x[:,0,:,:]*30+recon_x[:,1,:,:]*59+recon_x[:,2,:,:]*11)/100
	gray = gray.view(gray.shape[0],1,30,30)
	originloss = criterion(gray.float(),x.float()/255)
	return colorloss+originloss


def train(epoch_num):
	stacknn.train()
	for epoch in range(0,epoch_num):
		total_loss = 0
		i=0
		for data,target in dataloader:
			i=i+1
			optimizer.zero_grad()
			output = stacknn(data.float())
			loss = loss_func(output,data,target)
			loss.backward()
			total_loss += loss.item()
			optimizer.step()
		print('epoch:{}/{}: MSEloss:{:.8f}'.format(epoch, epoch_num,total_loss/i))
		if epoch%10 == 0:
			vitest()


recon = getrecon() #(1044, 1, 30, 30)
raw = getraw() #(1044, 3, 96, 96)

traindata=TensorDataset(torch.from_numpy(recon),torch.from_numpy(raw))
dataloader = DataLoader(traindata, batch_size=batch_size, shuffle=True)
testset = DataLoader(traindata, batch_size=12, shuffle=True)

stacknn=StackNN()
stacknn.load_state_dict(torch.load('stacknn.pth'))
optimizer = optim.Adam(stacknn.parameters(),lr=learning_rate)
optimizer.load_state_dict(torch.load('stackopt.pth'))

train(epoch_num)
vitest()

torch.save(stacknn.state_dict(),'./stacknn.pth')
torch.save(optimizer.state_dict(),'./stackopt.pth')
