import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import Augmentor

from video_data import VideoData
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 9
def create_video_data_labels(interpolation_frames, noise_parameters, matrix_size, kernel_size=2):
    xml_folder = '../some_xml_file'
    broken_videos_path = 'broken_videos.txt'
    data = []
    labels = []
    min_data = 99
    NORM_THRESHOLD = 3.7
    broken_videos = open(broken_videos_path, 'w')
    for label, folder in enumerate(os.listdir(xml_folder)):
        if folder == '.DS_Store':
            continue
        # print('folder', folder)
        broken_videos.write(folder + '\n')
        for file in os.listdir(xml_folder + '/' + folder):
            if file == '.DS_Store':
                continue
            file_path = xml_folder + '/' + folder + '/' + file
            video_data = VideoData(interpolations_frames=interpolation_frames, noise_frames=noise_parameters, matrix_size=matrix_size)
            video_data.load_xml_file(file_path)
            matrix = video_data.get_matrices()
            for frame in matrix:
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                if np.linalg.norm(np.array(cv2.dilate(frame, kernel, iterations=1))) < NORM_THRESHOLD :
                    broken_videos.write(file + '\n')
                    continue
                data.append(cv2.dilate(frame, kernel, iterations=1))
                plt.imshow(data[-1], cmap='gray')
                plt.title("name = " + file)
                #plt.figure()
                plt.savefig('../not_augmented/' + file + '.png')
                plt.close()

                #print('Folder:', folder,'File:', file)
                #print(np.linalg.norm(np.array(data[-1])))
                labels.append(label)

                #augmented_data, label_augmented = data_augmentation(data[-1],labels[-1], file)
                #data.append(augmented_data)
                #labels.append(label_augmented)
            if matrix.shape[0] < min_data:
                min_data = matrix.shape[0]

            #for norm_threshold in NORM_THRESHOLDS:
            #if np.linalg.norm(np.array(matrix), ord=None, axis=None, keepdims=False) <= 13:
                    #print('Folder:', folder, str(norm_threshold),'File:', file)
                    #plt.imshow(data[-1], cmap='gray')
                    #plt.title("name = " + file + str(norm_threshold))
                    #plt.figure()
                    #plt.show()
                    #print('Folder:', folder,'File:', file)

                    #broken_videos.write(file + 'has frobenius norm less than' + str(norm_threshold))

            # if label == 2:
            #     plt.imshow(matrix[2], cmap='gray')
            #     plt.title(file_path)
            #     plt.show()
        print(folder, "folder done. Label =", label)
    X_augmented, y_augmented = data_augmentation(data, labels, BATCH_SIZE)
    data.append(X_augmented)
    labels.append(y_augmented)
    print("Smallest matrix size is", min_data)
    return np.array(data), np.array(labels)

def visualize_augmented_data(X_batch, y_batch):
        # create a grid of 3x3 images
        for i in range(0, 9):
            plt.subplot(330 + 1 + i)
            plt.imshow(X_batch[i].reshape(22, 32), cmap=plt.get_cmap('gray'))
            if y_batch[i] == 1:
             plt.title('Lprev')
            if y_batch[i] == 2:
             plt.title('StartStop')
        # show the plot
        plt.show()
        #plt.savefig('../augmented/' + '.png')
        #plt.close()

def data_augmentation(data, labels, batch_size):
    data = np.array(data)
    #for X_batch, y_batch in data.flow(data, labels, batch_size=3):
        # create a grid of 3x3 images
    #    for i in range(0, 9):
    #        plt.subplot(330 + 1 + i)
    #        plt.imshow(X_batch[i].reshape(22, 32), cmap=plt.get_cmap('gray'))
        # show the plot
    #    plt.show()
    X_train = data.reshape(data.shape[0], 22, 32, 1)
    y_train = labels
    X_train = X_train.astype('float32')
    datagen = ImageDataGenerator(zoom_range=[0.99, 1.0],
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=0.1)
    # fit parameters from data
    datagen.fit(X_train)
    #Initialize the list of the output (the augmented data)
    X = []
    y = []
    # configure batch size and retrieve one batch of images
    for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):
        for i in range(0,9):
            X.append(X_batch[i].reshape(22, 32).tolist())
            y.append(y_batch[i].tolist())

        #visualize_augmented_data(X_batch, y_batch)
        break
    #X = np.array(X)
    #y = np.array(y)
    return X, y


def load_video_data_labels(interpolation_frames, noise_parameters, matrix_size=32):
    path = 'interpolation_' + str(interpolation_frames) + '_noise_' + str(
        noise_parameters) + '_matrix_size_' + str(matrix_size) + '.pkl'
    try:
        with open('../video_data_models/data_' + path, 'rb') as model:
            video_data = pickle.load(model)

        with open('../video_data_models/labels_' + path, 'rb') as model:
            video_labels = pickle.load(model)

        print("Video data and labels loaded from file")

    except:
        print("Failed to load video data or labels, create it")
        video_data, video_labels = create_video_data_labels(interpolation_frames, noise_parameters, matrix_size)
        with open('../video_data_models/data_' + path, 'wb') as output:
            pickle.dump(video_data, output)
        with open('../video_data_models/labels_' + path, 'wb') as output:
            pickle.dump(video_labels, output)

        print("Video data and labels created and saved")

    return video_data, video_labels


# print('CUDA is' + (' ' if torch.cuda.is_available() else ' not ') + 'available')
# data, labels = create_video_data_labels(7, 2, 32)
#
# # print("Data shape", data.shape)
# # print("labels shape", labels.shape)
# indexes = [i for i in range(len(labels))]
# np.random.shuffle(indexes)
#
# for i in indexes:
#     plt.imshow(data[i], cmap='gray')
#     plt.title("label = " + str(labels[i]))
#     plt.figure()
#     plt.show()

data, labels = create_video_data_labels(7, 2, 32)

for i in range(0, len(data)-1):
    #plt.subplot(1, len(data), i+1)
    plt.imshow(data[i].reshape(22, 32), cmap=plt.get_cmap('gray'))
    if labels[i] == 1:
        plt.title('Lprev')
    if labels[i] == 2:
        plt.title('StartStop')
    plt.show()
# show the plot
#plt.show()