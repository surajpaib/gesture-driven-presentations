import torch.nn.functional as F

from ml.autoencoder_new import *
from torch import optim


def get_latent_space_data(img_size=32, batch_size=32):
    """
    Return the latent space representation of the data
    :param img_size: pixels in width of the image
    :param batch_size: batch size for the dataloader
    :return: number of classes, dataloader
    """
    train_data, train_labels, dataloader = load_train_data(img_size, batch_size)

    n_classes = len(np.unique(train_labels))
    autoencoder = Autoencoder(train_data.shape[1] * train_data.shape[2])
    train_model()
    # print("Model trained")
    autoencoder.load_state()
    torch.device("cuda")
    latent_space = []
    for pos, image in enumerate(train_data):
        latent_space.append(autoencoder.get_latent_space(torch.from_numpy(image.reshape(-1))).data.numpy())

    target = np.array(train_labels)
    l_space = torch.from_numpy(np.array(latent_space))
    # Data Loader for easy mini-batch return in training, load the Dataset from the numpy arrays
    my_dataset = TensorDataset(l_space, torch.from_numpy(target).type(torch.LongTensor))
    dataloader = DataLoader(my_dataset, batch_size=batch_size)  # transform Dataset into a Dataloader

    return n_classes, dataloader


class Classifier(nn.Module):
    def __init__(self, n_classes):
        super(Classifier, self).__init__()
        # Input channel 1 because its gray scale
        self.layer = nn.Sequential(
                            nn.Linear(latent_space_dim, n_classes),
                            nn.ReLU())


    def forward(self, x):
        return self.layer(x)


def train_classifier():
    batch_size = 32
    # Load data
    n_classes, dataloader = get_latent_space_data(batch_size=batch_size)

    learning_rate = 0.01
    classifier = Classifier(4)
    epochs = 10000
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=learning_rate, weight_decay=1e-6, momentum=0.9, nesterov=True)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        i = 0
        for inputs, labels in dataloader:
            # 1. forward propagation
            output = classifier(inputs)

            # 2. loss calculation
            loss = loss_function(output, labels)

            # 3. backward propagation
            loss.backward()

            # 4. weight optimization
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:  # print every 200 mini-batches
                print('[%d, %5d] loss: %.4f' %
                      (epoch + 1, i + 1, loss / 200))
            i += 1

    print('Finished Training')


train_classifier()
print("HELLO")
