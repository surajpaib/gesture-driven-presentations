import torch
import torch.nn as nn
import torch.utils.data as utils
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from video_processing.load_data import *

# torch.manual_seed(1)    # reproducible
device = torch.device("cuda")
print("Using device", device)

# Hyper Parameters
EPOCH = 1000
BATCH_SIZE = 64
LR = 0.005  # learning rate
N_TEST_IMG = 10
LATENT_SPACE = 200

# Autoencoder does not have labels
train_data, train_labels = create_video_data_labels(7, 2, 32)

# Shuffle both arrays the same way
p = np.random.permutation(len(train_data))
train_data, train_labels = train_data[p], train_labels[p]

# np.random.shuffle(train_data)
# train_data = train_data[:, :-20, 15:]


# Transform to tensor
train_data_tensor = torch.from_numpy(train_data)
train_labels_tensor = torch.from_numpy(train_labels)

# Data Loader for easy mini-batch return in training, load the Dataset from the numpy arrays
my_dataset = utils.TensorDataset(train_data_tensor, train_labels_tensor)  # create your Dataset
train_loader = utils.DataLoader(my_dataset, batch_size=BATCH_SIZE)  # transform Dataset into a Dataloader


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(train_data.shape[1] * train_data.shape[2], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, LATENT_SPACE),
        )

        self.decoder = nn.Sequential(
            nn.Linear(LATENT_SPACE, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, train_data.shape[1] * train_data.shape[2]),
            nn.Sigmoid(),  # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = AutoEncoder(LATENT_SPACE).cuda()

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

# initialize figure
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()  # continuously plot

# original data (first row) for viewing
view_data = train_data_tensor[:N_TEST_IMG].contiguous().view(-1, train_data.shape[1] * train_data.shape[2]).type(
    torch.FloatTensor) / 255.
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (train_data.shape[1], train_data.shape[2])), cmap='gray')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())

for epoch in range(EPOCH):
    for step, (x, b_label) in enumerate(train_loader):
        b_x = x.view(-1, train_data.shape[1] * train_data.shape[2])  # batch x, shape (batch, 28*28)
        b_y = x.view(-1, train_data.shape[1] * train_data.shape[2])  # batch y, shape (batch, 28*28)

        encoded, decoded = autoencoder(b_x.cuda())

        loss = loss_func(decoded, b_y.cuda()) * 1000  # mean square error
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if step % 100 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy())

            # plotting decoded image (second row)
            _, decoded_data = autoencoder(view_data.cuda())
            for i in range(N_TEST_IMG):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.data.cpu().numpy()[i], (train_data.shape[1], train_data.shape[2])),
                               cmap='gray')
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())
            plt.draw()
            plt.pause(0.05)

plt.ioff()
# plt.show()

# visualize in 3D plot
view_data = train_data_tensor[:200].view(-1, train_data.shape[1] * train_data.shape[2]).type(
    torch.FloatTensor) / 255.
encoded_data, _ = autoencoder(view_data.cuda())
fig = plt.figure(2)
ax = Axes3D(fig)
X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
values = train_labels_tensor[:200].numpy()
for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255 * s / 9))
    ax.text(x, y, z, s, backgroundcolor=c)
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()
