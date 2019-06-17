from __future__ import annotations
import os

import numpy as np
from scipy import signal

from video_data import VideoData

class CorrelationClassifier:
    def __init__(self, dataset_path, interpolations_frames=4, noise_frames=2, matrix_size=64, confidence_threshold=0.5):

        # Settings to be passed to VideoData objects. Should probably match what is used for
        # the runtime VideoData.
        self.noise_frames = noise_frames
        self.interpolation_frames = interpolations_frames
        self.confidence_threshold = 0.5
        self.matrix_size = 64

        # Load the dataset (folders of XML files).
        self._load_dataset(dataset_path)

    def _load_dataset(self, dataset_path):
        """
        In dataset_path, considers each subfolder as representing one gesture. Loads all XML
        files found for each gesture and stores a VideoData object for each file.
        """

        # First just crawl the directory to see what gestures are available.
        self._gesture_labels = []
        self._dataset = {}
        self._flattened_matrices = {}
        for filename in os.listdir(dataset_path):
            if os.path.isdir(os.path.join(dataset_path, filename)):
                self._gesture_labels.append(filename)
                self._flattened_matrices[filename] = []
                self._dataset[filename] = []

        # Now, for each gesture, load the XML data.
        for label in self._gesture_labels:
            gesture_path = os.path.join(dataset_path, label)
            for filename in os.listdir(gesture_path):
                if filename.endswith('.xml'):
                    video_data = VideoData(self.interpolation_frames, noise_frames=self.noise_frames, \
                        matrix_size=self.matrix_size, confidence_threshold=self.confidence_threshold)
                    self._dataset[label].append(video_data)
                    self._flattened_matrices[label].append(video_data.get_flattened_matrix())

    def classify(self, runtime_matrix):
        """
        For now, tries to classify the given runtime matrix by performing the cross-correlation
        with all the flattened training matrices and taking the gesture for which the average
        peak-center distance is lowest.
        """

        lowest_distance = np.inf
        lowest_label = None

        for label in self._gesture_labels:
            flattened_matrices = self._flattened_matrices[label]
            peak_center_distances = []
            for other_matrix in flattened_matrices:
                corr = signal.correlate2d(runtime_matrix, other_matrix)
                # TODO: use fft to somehow speed this up. R_xy = ifft(fft(x,N) * conj(fft(y,N))) ??
                peak_index = np.unravel_index(np.argmax(corr, axis=None), corr.shape)
                distance = np.sqrt(np.square(peak_index[0] - 62) + np.square(peak_index[1] - 62))
                peak_center_distances.append(distance)

            avg_distance = np.average(peak_center_distances)
            if avg_distance < lowest_distance:
                lowest_distance = avg_distance
                lowest_label = label

        return lowest_label, lowest_distance
