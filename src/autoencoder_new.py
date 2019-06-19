import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST

from load_data import *

if not os.path.exists('./autoencoder_img'):
    os.mkdir('./autoencoder_img')

def to_img(x):
    #x = 0.5 * (x + 1)
    #x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 17, 32)
    # x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 32
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# dataset = MNIST('./data_MNIST', transform=img_transform)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Autoencoder does not have labels
train_data, train_labels = load_video_data_labels(7, 2, 32)
np.random.shuffle(train_data)

# Transform to tensor
train_data_tensor = torch.from_numpy(train_data)
train_labels_tensor = torch.from_numpy(train_labels)
#train_data_tensor = img_transform(train_data)
#train_labels_tensor = img_transform(train_labels)

# Data Loader for easy mini-batch return in training, load the Dataset from the numpy arrays
my_dataset = TensorDataset(train_data_tensor, train_labels_tensor)  # create your Dataset
dataloader = DataLoader(my_dataset, batch_size=batch_size)  # transform Dataset into a Dataloader



class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
           nn.Linear(train_data.shape[1] * train_data.shape[2], 128),
            # nn.Linear(28* 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            # nn.ReLU(True), nn.Linear(128,28 * 28), nn.Tanh())
            nn.ReLU(True), nn.Linear(128,train_data.shape[1] * train_data.shape[2]), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data.cpu().numpy()))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './autoencoder_img/image_{}_output.png'.format(epoch))

        input_pic = to_img(img.cpu().data)
        save_image(input_pic, './autoencoder_img/image_{}_input.png'.format(epoch))

torch.save(model.state_dict(), './sim_autoencoder.pth')