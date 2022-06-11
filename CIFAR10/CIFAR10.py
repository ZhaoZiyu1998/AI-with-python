import torch
import torchvision as tv
import torchvision.transforms as tf
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.datasets as datasets
def run():
   # code goes here
    normalize = tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_set=tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=tf.Compose([tf.RandomCrop(32, padding=4), tf.RandomHorizontalFlip(), tf.ToTensor(), normalize]))
    test_set=tv.datasets.CIFAR10(root='./data', train=False, download=True, transform=tf.Compose([tf.ToTensor(), normalize]))
    train_loader=torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    test_loader=torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs=10


    class ConvolutionModel(nn.Module):

        def __init__(self):
            super(ConvolutionModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(8*8*64, 120)
            self.fc2 = nn.Linear(120, 60)
            self.fc3 = nn.Linear(60, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))

            x = x.view(-1, 64 * 8 * 8)
            x = F.relu(self.fc1(x))

            x = F.relu(self.fc2(x))
            x = self.fc3(x)

            return x
    category_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    model = ConvolutionModel()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
           # y_hat = torch.max(outputs, 1)[1]
            #correct = (y_hat == labels).sum().item()
            #print('Epoch: {}/{} | Batch: {}/{} | Loss: {} | Accuracy: {}%'.format(epoch+1, epochs, i+1, len(train_loader), loss.item(), 100*correct/len(images)))
            if (i+1) % 100 == 0:
                print('Epoch: {}/{} | Batch: {}/{} | Loss: {}'.format(epoch+1, epochs, i+1, len(train_loader), loss.item()))
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the model on the test images: {}%'.format(100 * correct / total))
    torch.save(model.state_dict(), 'cifar10_model.pt')
if __name__ == '__main__':
    run()