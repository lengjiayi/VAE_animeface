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
from PIL import Image


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



#selest two faces randomly and generate faces using interval vector
def fixin():
	bdx = random.randint(0,len(codes))
	bdx2 = random.randint(0,len(codes))
	sample1 = codes[bdx]
	sample2 = codes[bdx2]
	gcodeset = []
	dist = sample2-sample1
	for i in range(1,10):
		newface=sample1+dist/8*(i-1)
		newface = vae.decoder(torch.from_numpy(newface).float())
		plt.subplot(1,11,i+1)
		plt.xticks([])
		plt.yticks([])
		plt.imshow(newface.detach().numpy().reshape(30,30),cmap='gray')
	face1 = sample[bdx].reshape(30,30)
	face2 = sample[bdx2].reshape(30,30)
	plt.subplot(1,11,1)
	plt.xticks([])
	plt.yticks([])
	plt.imshow(face1,cmap='gray')
	plt.subplot(1,11,11)
	plt.xticks([])
	plt.yticks([])
	plt.imshow(face2,cmap='gray')
	plt.subplots_adjust(wspace=0,hspace=0,left=None,right=None,bottom=None,top=None)
	plt.show()



#compare 25 raw faces and reconstrain faces
def visgenerate():
	rfset=[]
	for i in range(25):
		rf = np.random.normal(0,3,10)
		newface = vae.decoder(torch.from_numpy(rf).float())
		rfset.append(newface.detach().numpy().reshape(30,30))
	for i in range(1,26):
		plt.subplot(5,5,i)
		plt.xticks([])
		plt.yticks([])
		plt.imshow(rfset[i-1],cmap='gray')
	plt.subplots_adjust(wspace=0,hspace=0,left=None,right=None,bottom=None,top=None)
	plt.show()


#generate 'num' faces randomly
def genran(num):
	for i in range(num):
		rf = np.random.normal(0,1,10)
		newface = vae.decoder(torch.from_numpy(rf).float())
		newface = newface.detach().numpy().reshape(30,30)
		img = Image.fromarray(np.uint8(newface)).resize((100,100))
		img.save('./RandomGen/'+str(i)+'.png')



vae=VAE()
vae.load_state_dict(torch.load('vaeg.pth'))

sample = getdata()
testset = list(DataLoader(sample, batch_size=1, shuffle=False))
codes=[]
muset = []
for x in testset:
	tmp=vae.encoder(x.float())
	logvar=vae.fcvar(tmp)
	mu=vae.fcmu(tmp)
	muset.append(mu.detach().numpy())
	z=vae.reparameterize(mu,logvar)
	codes.append(z.detach().numpy()[0])
codes = np.array(codes)
#code space min edge
#[-3.0761225 -2.8624005 -3.2875185 -3.4103947 -2.4884589 -2.5605006 -3.231966  -3.0437517 -2.5869646 -3.1095586]
minspace = codes.min(axis=0)
#code space max edge
#[3.6045144 2.81057   3.354098  4.6846285 2.9751196 3.0764222 2.7891662 3.000442  2.3526127 2.9774058]
maxspace = codes.max(axis=0)
#np.random.seed(0)
#visgenerate()
#genran(100)
fixin()