import os
import pickle

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from video_processing.video_data import VideoData
from config import CONFIG


def create_video_data_labels(interpolation_frames=CONFIG["interpolation_frames"], noise_parameters=CONFIG["noise_frames"], used_keypoints=CONFIG["used_keypoints"], matrix_size=CONFIG["matrix_size"], use_dilation=CONFIG["use_dilation"], kernel_size=CONFIG["kernel_size"]):
    xml_folder = os.path.dirname(os.path.realpath(__file__)).split("src")[0].replace("\\", "/") + CONFIG["xml_files_path"]
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
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    data.append(cv2.dilate(frame, kernel, iterations=1))
                else:
                    data.append(frame)

                labels.append(label)
            if matrix.shape[0] < min_data:
                min_data = matrix.shape[0]

        print(folder, "folder done. Label =", label)
    print("Smallest matrix size is", min_data)
    return np.array(data), np.array(labels)


def load_video_data_labels(interpolation_frames=CONFIG["interpolation_frames"], noise_parameters=CONFIG["noise_frames"], used_keypoints=CONFIG["used_keypoints"], matrix_size=CONFIG["matrix_size"]):
    path = 'interpolation_' + str(interpolation_frames) + '_noise_' + str(
        noise_parameters) + '_matrix_size_' + str(matrix_size) + '.pkl'

    video_data_path = os.path.dirname(os.path.realpath(__file__)).split("src")[0].replace("\\","/") + "video_data_models/"
    try:
        with open(video_data_path + 'data_' + path, 'rb') as model:
            video_data = pickle.load(model)

        with open(video_data_path + 'labels_' + path, 'rb') as model:
            video_labels = pickle.load(model)

        print("Video data and labels loaded from file")

    except:
        print("Failed to load video data or labels, create it")
        video_data, video_labels = create_video_data_labels(interpolation_frames, noise_parameters, used_keypoints, matrix_size)
        with open(video_data_path + 'data_' + path, 'wb') as output:
            pickle.dump(video_data, output)
        with open(video_data_path + 'labels_' + path, 'wb') as output:
            pickle.dump(video_labels, output)

        print("Video data and labels created and saved")

    return video_data, video_labels

if __name__ == "__main__":
    data, labels = create_video_data_labels(interpolation_frames=4, noise_parameters=2, matrix_size=64)
    indexes = [i for i in range(len(labels))]
    np.random.shuffle(indexes)
    
    for i in indexes:
        plt.imshow(data[i], cmap='gray')
        plt.title("label = " + str(labels[i]))
        plt.figure()
        plt.show()

