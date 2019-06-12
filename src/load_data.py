import os
import pickle

import cv2
import numpy as np

from video_data import VideoData


def create_video_data_labels(interpolation_frames, noise_parameters, matrix_size, kernel_size=2):
    xml_folder = '../xml_files'
    data = []
    labels = []
    min_data = 99
    for label, folder in enumerate(os.listdir(xml_folder)):
        # print('folder', folder)
        for file in os.listdir(xml_folder + '/' + folder):
            file_path = xml_folder + '/' + folder + '/' + file
            video_data = VideoData(interpolations_frames=interpolation_frames, noise_frames=noise_parameters)
            video_data.load_xml_file(file_path)
            matrix = video_data.generate_matrices(matrix_size)
            for frame in matrix:
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                data.append(cv2.dilate(frame, kernel, iterations=1))
                labels.append(label)
            if matrix.shape[0] < min_data:
                min_data = matrix.shape[0]
        print(folder, "folder done. Label =", label)
    print("Smallest matrix size is", min_data)
    return np.array(data), np.array(labels)


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
# data, labels = create_video_data_label(7, 2)
# print("Data shape", data.shape)
# print("labels shape", labels.shape)
#
# import matplotlib.pyplot as plt
# for i in range(len(labels)):
#     plt.imshow(data[i], cmap='gray')
#     plt.title("label = " + str(labels[i]))
#     plt.figure()
#     if i % 300 == 0:
#         plt.show()
