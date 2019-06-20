import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import Augmentor

from video_data import VideoData


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
                augmented_data, label_augmented = data_augmentation(data[-1],labels[-1], file)
                data.append(augmented_data)
                labels.append(label_augmented)

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
    print("Smallest matrix size is", min_data)
    return np.array(data), np.array(labels)

def data_augmentation(data, label, file):
    p = Augmentor.Pipeline(data)
    p.zoom(probability=0.5, min_factor=1.1, max_factor=1.3)
    p.zoom_random(probability=0.5, percentage_area=0.2)
    p.random_distortion(probability=0.6, grid_width=4, grid_height=4, magnitude=8)
    augmented_data, label = p.sample(50)
    #for index in range(len)
    plt.imshow(augmented_data[-1], cmap='gray')
    plt.title("name = " + file)
    # plt.figure()
    plt.savefig('../augmented/' + file + '.png')
    plt.close()
    return augmented_data, label

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