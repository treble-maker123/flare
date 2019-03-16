########################################################
# Branched off of models/pre-trained_2d/resnet/resnet.py
# This is a simple implementation of pretrained resnet 
# on 2D slices.
# Didn't want to delete anything from above file while
# building this code so created new file here.
# Implemented dataset loader etc and got resnet to work
# on that.  3/16/2019
#######################################################

import os
import cv2
import torch
import pickle
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark=True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

lrate = 0.001
num_epochs = 6
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,3)
model.avgpool = nn.AdaptiveAvgPool2d(1)

#send to GPU
model.to(device)

#define loss function + optimizer
optimizer = optim.Adam(model.parameters(),lr=lrate)
criterion = nn.CrossEntropyLoss()

class ADNIDataset2D(Dataset):
	def __init__(self, mode="train", data_ver="presliced"):
		self.mode = mode
		self.data_ver = data_ver
		trans = [T.ToTensor()]
		self.transform = T.Compose(trans)
         	if data_ver == "presliced":
			self.subsample_path = "/mnt/nfs/work1/mfiterau/ADNI_data/slice_subsample_no_seg/coronal_skullstrip"
                	self.list_of_subjectdir = os.listdir(self.subsample_path)
		if data_ver == "liveslice":
			with open('outputs/normalized_mapping.pickle', 'rb') as f:
                         	self.data = pickle.load(f)
        
    	def __len__(self):
	        if self.mode == "train" and self.data_ver=="presliced":
			return 300 * 4/5
	    	elif self.mode == "val" and self.data_ver=="presliced":
			return 300 * 1/5
            	elif self.mode == "train" and self.data_ver=="liveslice":
                	return (self.data).size * 4/5
            	elif self.mode == "val" and self.data_ver=="liveslice":
                	return (self.data).size * 1/5
	
        def __getitem__(self, idx):
		if self.data_ver=="presliced":
	        	img, label = self._get_item_presliced_helper()
            	elif self.data_ver=="liveslice":
                	img, label = self._get_item_livesslice_helper()
            	return img, label

        def _get_item_presliced_helper(self):
		if self.mode == "train":
		    list_of_subjectdir = self.list_of_subjectdir[:240]
		elif self.mode == "val":
		    list_of_subjectdir = self.list_of_subjectdir[-60:]
		subject_path = random.choice(list_of_subjectdir)
		img = cv2.imread(self.subsample_path+"/"+subject_path+"/normalized_seg_33.tiff")
		if img is not None:
		    img = img[:,:,[2,1,0]]
		    # Just crop out the white paths, its consistently the same place..
                    img = img[50:-40, 25:-35, :]
		    img = cv2.resize(img, (256, 256))
		    img = self.transform(img)
		    label = subject_path.strip(".tiff").split("_")[-1]
		    label = 2 if (label=="AD") else (1 if (label=="MCI") else 0)
                    return img, label
		else:
		    print my_dir+"/"+img_path+"/normalized_seg_33.tiff"
        
     	def _get_item_liveslice_helper(self):
           	pass

train_dataset = ADNIDataset2D(mode="train")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataset = ADNIDataset2D(mode="val")
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

for epoch in range(num_epochs):
        running_loss = 0.0
        total_train = 0
        train_correct = 0
	for num_iter, (x, y) in enumerate(train_loader):
		x, y = x.to(device), y.to(device)
		optimizer.zero_grad()
		pred = model(x)
		loss = criterion(pred, y)
		loss.backward()
		optimizer.step()

		# count correct predictions in training set
                total_train += y.size(0)
                _, pred = torch.max(pred.data,1)
                train_correct += (pred == y).sum().item()
		
		#print epoch, num_iter, loss
	print "Train Accuracy", epoch, train_correct, total_train

total_val = 0
val_correct = 0
for num_iter, (x, y) in enumerate(val_loader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_val += y.size(0)
        _, pred = torch.max(pred.data,1)
        val_correct += (pred == y).sum().item()
print "Val Accuracy", epoch, val_correct, total_val
