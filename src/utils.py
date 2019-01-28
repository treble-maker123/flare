import os
import nibabel as nib
import numpy as np
import torch
import pickle
import skimage.transform

def load_img(path, view='axial'):
    if(path[-3:] == 'nii'):
        image = nib.load(path).get_fdata()
        image = skimage.transform.resize(image, (40,120,128))
        return image
    elif(path[-3:] == 'npz'):
        try:
            with open(path,'rb') as f:
                image = pickle.load(f)
                return image
        except IOError:
            print("image does not exist in path")

def create_dirs(paths):
    for path in paths:
        if os.path.exists(path)==False:
            os.mkdir(path)

def one_hot(labels, C=5):
    N = len(labels)
    labels_onehot = torch.zeros([N, C], dtype=torch.float)
    labels = torch.squeeze(labels) - 1
    labels_onehot[torch.arange(N), labels.type(torch.long)] = 1
    return labels_onehot
