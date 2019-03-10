import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.benchmark=True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

#JOIE: routine to train and test densenet on CIFAR images. model is then saved.


#normalize = transforms.Normalize(mean=[0.485,0.456,.406],std=[0.229,0.224,0.225])

transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5, 0.5, 0.5))])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                        download=True, transform=transform)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, 
                                            shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                        shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
            'ship', 'truck')



#load pre-trained densenet
densenet = models.densenet161(pretrained=True)

#send to GPU
densenet.to(device)

#define loss function + optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(densenet.parameters(),lr=0.001, momentum=0.9)

#train the dense net

print("Training the densenet")
for epoch in range(1):
    print("Epoch", epoch)
    running_loss = 0.0
    for i,data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = densenet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % 
                (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    
print('Finished training')

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images,labels = images.to(device), labels.to(device)
        output = densenet(images)
        _, predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy on 10000 test images: %d %%' % (100*correct / total))

