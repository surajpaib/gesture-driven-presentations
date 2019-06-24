import os
import pickle

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import CONFIG
from ml.autoencoder import Autoencoder
from ml.classifier import Classifier
from video_processing.video_data import VideoData


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
    autoencoder.load_state()

    autoencoder = train_autoencoder()
    # print("Model trained")

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


def to_img(x):
    # x = 0.5 * (x + 1)
    # x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 22, 32)
    # x = x.view(x.size(0), 1, 28, 28)
    return x


def load_train_data(img_size=CONFIG["matrix_size"], batch_size=CONFIG["batch_size"],
                    used_keypoints=CONFIG["used_keypoints"], interpolation_frames=CONFIG["interpolation_frames"],
                    noise_frames=CONFIG["noise_frames"]):
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


def train_autoencoder(img_size=CONFIG["matrix_size"], batch_size=CONFIG["batch_size"], num_epochs=CONFIG["num_epochs"],
                      used_keypoints=CONFIG["used_keypoints"]):
    from torchvision.utils import save_image

    train_data, train_labels, dataloader = load_train_data(img_size, batch_size, used_keypoints)
    learning_rate = CONFIG["learning_rate"]
    autoencoder = Autoencoder(train_data.shape[1] * train_data.shape[2]).cuda()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        autoencoder.parameters(), lr=learning_rate, weight_decay=CONFIG["weight_decay"])
    for epoch in range(num_epochs):
        for data in dataloader:
            img, _ = data
            img = img.view(img.size(0), -1)
            img = torch.Variable(img).cuda()
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


def train_classifier():
    batch_size = 32
    # Load data
    n_classes, dataloader = get_latent_space_data(batch_size=batch_size)

    learning_rate = 0.001
    classifier = Classifier(4)
    epochs = 500
    loss_function = torch.nn.CrossEntropyLoss()
    # optimizer = optim.SGD(classifier.parameters(), lr=learning_rate, weight_decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = torch.optim.Adam(
        classifier.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_epochs = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        loss_history = []
        for inputs, labels in dataloader:
            # 1. forward propagation
            output = classifier(inputs)

            # 2. loss calculation
            loss = loss_function(output, labels)

            # 3. backward propagation
            loss.backward()

            # 4. weight optimization
            optimizer.step()

            loss_history.append(loss.item())

        epoch_loss = np.average(loss_history)
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, epochs, epoch_loss))
        loss_epochs.append(epoch_loss)

    import matplotlib.pyplot as plt
    plt.plot(loss_epochs)
    plt.title("Loss history")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
    classifier.save_state()
    print('Finished Training')

    return classifier


def create_video_data_labels(interpolation_frames=CONFIG["interpolation_frames"],
                             noise_parameters=CONFIG["noise_frames"], used_keypoints=CONFIG["used_keypoints"],
                             matrix_size=CONFIG["matrix_size"], use_dilation=CONFIG["use_dilation"],
                             kernel_size=CONFIG["kernel_size"]):
    """
    Load the xmls files and create images using interpolation and the labels assigned to each of the images.

    :param interpolation_frames: Number of frames to use for the interpolations
    :param noise_parameters: Number of frames that are considered to be noise
    :param used_keypoints: keypoints to use when loading the frames
    :param matrix_size: Size of the images returned
    :param use_dilation: Flag to use dilation on the images
    :param kernel_size: Size of the kernel for the dilation
    :return: video data and labels
    """
    xml_folder = os.path.dirname(os.path.realpath(__file__)).split("src")[0].replace("\\", "/") + CONFIG[
        "xml_files_path"]
    data = []
    labels = []
    min_data = 99
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    for label, folder in enumerate(os.listdir(xml_folder)):
        for file in os.listdir(xml_folder + '/' + folder):
            file_path = xml_folder + '/' + folder + '/' + file
            video_data = VideoData(interpolations_frames=interpolation_frames, matrix_size=matrix_size,
                                   used_keypoints=used_keypoints, noise_frames=noise_parameters)
            video_data.load_xml_file(file_path)
            matrix = video_data.get_matrices()
            for frame in matrix:

                # Apply dilation if enabled.
                if use_dilation:
                    data.append(cv2.dilate(frame, kernel, iterations=1))
                else:
                    data.append(frame)

                labels.append(label)
            if matrix.shape[0] < min_data:
                min_data = matrix.shape[0]

        print(folder, "folder done. Label =", label)
    print("Smallest matrix size is", min_data)
    return np.array(data), np.array(labels)


def load_video_data_labels(interpolation_frames=CONFIG["interpolation_frames"], noise_parameters=CONFIG["noise_frames"],
                           used_keypoints=CONFIG["used_keypoints"], matrix_size=CONFIG["matrix_size"]):
    """
    Load the images and if they are not saved as pickle files then call the function to create them
    :param interpolation_frames: Number of frames to use for the interpolations
    :param noise_parameters: Number of frames that are considered to be noise
    :param used_keypoints: keypoints to use when loading the frames
    :param matrix_size: Size of the images returned
    :return: video data and labels
    """
    path = 'interpolation_' + str(interpolation_frames) + '_noise_' + str(
        noise_parameters) + '_matrix_size_' + str(matrix_size) + '.pkl'

    video_data_path = os.path.dirname(os.path.realpath(__file__)).split("src")[0].replace("\\",
                                                                                          "/") + "video_data_models/"
    try:
        with open(video_data_path + 'data_' + path, 'rb') as model:
            video_data = pickle.load(model)

        with open(video_data_path + 'labels_' + path, 'rb') as model:
            video_labels = pickle.load(model)

        print("Video data and labels loaded from file")

    except:
        print("Failed to load video data or labels, create it")
        video_data, video_labels = create_video_data_labels(interpolation_frames, noise_parameters, used_keypoints,
                                                            matrix_size)
        with open(video_data_path + 'data_' + path, 'wb') as output:
            pickle.dump(video_data, output)
        with open(video_data_path + 'labels_' + path, 'wb') as output:
            pickle.dump(video_labels, output)

        print("Video data and labels created and saved")

    return video_data, video_labels
