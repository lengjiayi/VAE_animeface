import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import csv
import random


learning_rate = 0.0001
epoch_num = 1000
batch_size = 100

criterion = nn.MSELoss()


def getdata():
	sample = []
	text = open('gray.txt','r',encoding='utf-8')
	row = csv.reader(text, delimiter=',')
	for r in row:
		pict = []
		for pix in r:
			if pix:
#				tmp = np.zeros(143)
#				tmp[int(pix)]=1
#				pict.append(tmp)
				pict.append(int(pix))
		sample.append(pict)
	sample = np.array(sample)
	return sample


class VAE(nn.Module):
	def __init__(self):
		super(VAE, self).__init__()
		self.ZDim=10
		self.encoder = nn.Sequential(
#			nn.Linear(30*30*143,30*30),
#			nn.ReLU(True),
			nn.Linear(30*30,512),
			nn.ReLU(True),
			nn.Linear(512,96),
			nn.ReLU(True),
			nn.Linear(96,25),
			)
		self.fcmu=nn.Linear(25,self.ZDim)
		self.fcvar=nn.Linear(25,self.ZDim)
		self.decoder = nn.Sequential(
			nn.Linear(self.ZDim,25),
			nn.ReLU(True),
			nn.Linear(25,96),
			nn.ReLU(True),
			nn.Linear(96,512),
			nn.ReLU(True),
			nn.Linear(512,30*30),
#			nn.ReLU(True),
#			nn.Linear(900,30*30*143),
#			nn.Sigmoid()
			)
	def reparameterize(self, mu, logvar):
		#z=exp(loavar)*eps+mu
		eps = torch.randn(mu.size(0),mu.size(1))
		z=mu+eps*torch.exp(logvar/2)
		return z

	def forward(self, x):
		x=self.encoder(x)
		logvar=self.fcvar(x)
		mu=self.fcmu(x)
		z=self.reparameterize(mu,logvar)
		#return reconstructed sample, mu and logvar
		return self.decoder(z),mu,logvar


def loss_func(recon_x, x, mu, logvar):
	BCE = criterion(recon_x.float(), x.float())
	#Minimize{1+logvar-(mu)^2-exp(logvar)}
	KLD=-0.5* torch.sum(1+logvar-mu.pow(2)-logvar.exp())
	return BCE+KLD



def vitest():
	norm = testset[0]
	norm = norm.detach().numpy()
	for i in range(1,19):
		test=norm[i-1]
		test=test.reshape(30,30)
		plt.subplot(6,6,i)
		plt.xticks([])
		plt.yticks([])
		plt.imshow(test,cmap='gray')
	norm,_,_ = vae(testset[0].float())
	norm = norm.detach().numpy()
	for i in range(1,19):
		test=norm[i-1]
		test=test.reshape(30,30)
		plt.subplot(6,6,i+18)
		plt.xticks([])
		plt.yticks([])
		plt.imshow(test,cmap='gray')
	mng = plt.get_current_fig_manager()
	mng.window.state('zoomed')
	plt.subplots_adjust(wspace=0,hspace=0,left=None,right=None,bottom=None,top=None)
	plt.show()





def train(epoch_num):
	vae.train()
	for epoch in range(0,epoch_num):
		total_loss = 0
		i=0
		for data in dataloader:
			i=i+1
			optimizer.zero_grad()
			(recon_x, mu, logvar) = vae(data.float())
			loss = loss_func(recon_x,data,mu,logvar)
			loss.backward()
			total_loss += loss.item()
			optimizer.step()
#			print('VAEloss:{:.8f}'.format(loss.item()))
		vitest()
		print('epoch:{}/{}: VAEloss:{:.8f}'.format(epoch, epoch_num,total_loss/i))



vae=VAE()
vae.load_state_dict(torch.load('vaeg.pth'))
optimizer = optim.Adam(vae.parameters(),lr=learning_rate)
optimizer.load_state_dict(torch.load('optg.pth'))

sample = getdata()
dataloader = DataLoader(sample, batch_size=batch_size, shuffle=False)
testset = list(DataLoader(sample, batch_size=18, shuffle=True))
train(epoch_num)
vitest()
#torch.save(vae.state_dict(),'./vaeg.pth')
#torch.save(optimizer.state_dict(),'./optg.pth')
