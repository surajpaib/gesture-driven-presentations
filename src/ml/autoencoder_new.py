import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image

from video_processing.load_data import *

if not os.path.exists('./autoencoder_img'):
    os.mkdir('./autoencoder_img')

latent_space_dim = 20


class Autoencoder(nn.Module):
    def __init__(self, train_data_shape):
        super(Autoencoder, self).__init__()
        self.path = '../autoencoder_' + str(train_data_shape) + '.pth'
        self.encoder = nn.Sequential(
            nn.Linear(train_data_shape, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, latent_space_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_space_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, train_data_shape),
            nn.Tanh())

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_latent_space(self, x):
        return self.encoder(x)

    def load_state(self):
        self.load_state_dict(torch.load(self.path))

    def save_state(self):
        torch.save(self.state_dict(), self.path)


def to_img(x):
    # x = 0.5 * (x + 1)
    # x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 22, 32)
    # x = x.view(x.size(0), 1, 28, 28)
    return x


def load_train_data(img_size=32, batch_size=32):
    # Autoencoder does not have labels
    train_data, train_labels = load_video_data_labels(7, 2, img_size, main=False)
    p = np.random.permutation(len(train_data))
    train_data, train_labels = train_data[p], train_labels[p]

    # Transform to tensor
    train_data_tensor = torch.from_numpy(train_data)
    train_labels_tensor = torch.from_numpy(train_labels)

    # Data Loader for easy mini-batch return in training, load the Dataset from the numpy arrays
    my_dataset = TensorDataset(train_data_tensor, train_labels_tensor)  # create your Dataset
    dataloader = DataLoader(my_dataset, batch_size=batch_size)  # transform Dataset into a Dataloader

    return train_data, train_labels, dataloader


def train_model(img_size=32, batch_size=32):
    train_data, train_labels, dataloader = load_train_data(img_size, batch_size)
    num_epochs = 5
    learning_rate = 1e-3
    autoencoder = Autoencoder(train_data.shape[1] * train_data.shape[2]).cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)
    for epoch in range(num_epochs):
        for data in dataloader:
            img, _ = data
            img = img.view(img.size(0), -1)
            img = Variable(img).cuda()
            # ===================forward=====================
            output = autoencoder(img)
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

    autoencoder.save_state()
    return autoencoder

# train_model()
