import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image

import sys
sys.path.append('./')
from video_processing.load_data import *
from config import CONFIG

if not os.path.exists(CONFIG["autoencoder_img_path"]):
    os.mkdir(CONFIG["autoencoder_img_path"])


class Autoencoder(nn.Module):
    def __init__(self, train_data_shape, latent_space_dim=CONFIG["latent_space_dim"]):
        super(Autoencoder, self).__init__()
        self.path = os.path.dirname(os.path.realpath(__file__)).split("src")[0].replace("\\", "/") + \
                    'autoencoder_' + str(train_data_shape) + '.pth'
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
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

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


def load_train_data(used_keypoints=CONFIG["used_keypoints"], img_size=CONFIG["matrix_size"], batch_size=CONFIG["batch_size"], interpolation_frames=CONFIG["interpolation_frames"], noise_frames=CONFIG["noise_frames"]):
    # Autoencoder does not have labels
    train_data, train_labels = load_video_data_labels(interpolation_frames, noise_frames, used_keypoints, img_size)
    p = np.random.permutation(len(train_data))
    train_data, train_labels = train_data[p], train_labels[p]

    # Transform to tensor
    train_data_tensor = torch.from_numpy(train_data)
    train_labels_tensor = torch.from_numpy(train_labels)

    # Data Loader for easy mini-batch return in training, load the Dataset from the numpy arrays
    my_dataset = TensorDataset(train_data_tensor, train_labels_tensor)  # create your Dataset
    dataloader = DataLoader(my_dataset, batch_size=batch_size)  # transform Dataset into a Dataloader

    return train_data, train_labels, dataloader

def train_model(img_size=CONFIG["matrix_size"], batch_size=CONFIG["batch_size"], num_epochs=CONFIG["num_epochs"], used_keypoints=CONFIG["used_keypoints"]):
    train_data, train_labels, dataloader = load_train_data(used_keypoints, img_size, batch_size)
    learning_rate = CONFIG["learning_rate"]
    autoencoder = Autoencoder(train_data.shape[1] * train_data.shape[2]).cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        autoencoder.parameters(), lr=learning_rate, weight_decay=CONFIG["weight_decay"])
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
        if epoch % 5 == 0:
            pic = to_img(output.cpu().data)
            save_image(pic, os.path.join(CONFIG["autoencoder_img_path"], 'image_{}_output.png'.format(epoch)))

            input_pic = to_img(img.cpu().data)
            save_image(input_pic, os.path.join(CONFIG["autoencoder_img_path"], 'image_{}_input.png'.format(epoch)))

    autoencoder.save_state()
    return autoencoder

def main():
    train_model()

if __name__== "__main__":
    main()
