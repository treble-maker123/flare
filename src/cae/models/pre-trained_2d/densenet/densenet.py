import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch import randperm
import os.path

torch.backends.cudnn.benchmark=True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

#JOIE: routine to train and test densenet on CIFAR images. model is then saved.


#normalize = transforms.Normalize(mean=[0.485,0.456,.406],std=[0.229,0.224,0.225])

transform = transforms.Compose(
                [transforms.ToTensor()])
                #transforms.Normalize((0.5,0.5,0.5),(0.5, 0.5, 0.5))])


cifartrain = torchvision.datasets.CIFAR10(root='../data', train=True, 
                                        download=True , transform=transform)

indices = randperm(50000)

if(os.path.isfile('../data/cifar10train')):
    with open('../data/cifar10train','r') as f:
        cifartrain = f.read()

if(os.path.isfile('../data/cifar10val')):
    with open('../data/cifar10val','r') as f:
        cifarval = f.read()



#trainset, valset = torch.utils.data.random_split(cifartrain, [40000,10000])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, 
                                            shuffle=True, num_workers=2)

valloader = torch.utils.data.DataLoader(valset, batch_size=128, 
                                            shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                        download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                        shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
            'ship', 'truck')


#load pre-trained densenet
densenet = models.densenet161(pretrained=True)
densenet.classifier = nn.Linear(2208, 10)

#send to GPU
densenet.to(device)

#define loss function + optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(densenet.parameters(),lr=0.001)

#For plotting later
val_acc_vals = []
train_acc_vals = []
train_loss_vals = []
val_loss_vals = []
epochs = []

#train the res net
print("Training the densenet")
for epoch in range(150):
    
    running_loss = 0.0
    total_train = 0
    train_correct = 0

    for data in trainloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = densenet(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # count correct predictions in training set
        total_train += labels.size(0)
        _, predicted = torch.max(outputs.data,1)
        train_correct += (predicted == labels).sum().item ()
    
    # calculate training_accuracy
    train_acc = 100*train_correct/total_train
    train_loss = running_loss/total_train

    print('[epoch %d] Train Loss: %.8f, Train Accuracy: %.8f' % 
        (epoch + 1, train_loss/total_train, train_acc))
   
    # validation 
    densenet.eval()
    total_val = 0
    correct = 0
    val_loss = 0
    val_acc = 0

    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device),labels.to(device)
            outputs = densenet(images)
            _, predicted = torch.max(outputs.data,1)
            total_val += labels.size(0)
            batch_loss = criterion(outputs, labels)
            val_loss += batch_loss.item()
            correct += (predicted == labels).sum().item()

    val_acc = 100*correct/total_val

    print('Validation Loss: %.5f, Validation Accuracy: %.5f' % 
        (val_loss/total_val,val_acc))

    if(epoch % 5 == 4):
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': densenet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                    }, './checkpoints/densenet_lr_0.01_cifar_epoch_{}'.format(epoch+1))

    densenet.train()
    
    # add loss and accuracy points to plot
    val_acc_vals.append(val_acc) 
    train_acc_vals.append(train_acc)
    train_loss_vals.append(train_loss)
    val_loss_vals.append(val_loss/total_val) 
    epochs.append(epoch)

print('Finished training')

"""
JOIE: generate graph consisting of two subgraphs: 
    top: val loss vs train loss 
    bottom: val ac vs train acc

"""

f = plt.figure(figsize=(10,10))

f.add_subplot(2,1,1)
plt.plot(epochs,train_loss_vals,label='train loss')
plt.plot(epochs,val_loss_vals,label='val loss')
plt.title('Loss and Accuracy over Epochs')
plt.legend()
plt.ylabel('loss')

f.add_subplot(2,1,2)
plt.plot(epochs,train_acc_vals,label='train acc')
plt.plot(epochs,val_acc_vals,label='val_acc')
plt.ylabel('acc')
plt.legend()

plt.savefig('./figures/densenet_lr_0.01_train_val_loss.png')

#begin testing
correct = 0
total = 0

densenet.eval()
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images,labels = images.to(device), labels.to(device)
        outputs = densenet(images)
        _, predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy on 10000 test images: %d %%' % (100*correct / total))
