# from __future__ import annotations
import os

import numpy as np
from scipy import signal

from video_processing.video_data import VideoData


class CorrelationClassifier:
    def __init__(self, dataset_path, used_keypoints, interpolations_frames=4, noise_frames=2, matrix_size=64, confidence_threshold=0.5):

        # Settings to be passed to VideoData objects. Should probably match what is used for
        # the runtime VideoData.
        self.noise_frames = noise_frames
        self.interpolation_frames = interpolations_frames
        self.confidence_threshold = 0.5
        self.matrix_size = 64
        self.used_keypoints = used_keypoints

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
                    video_data = VideoData(self.interpolation_frames, matrix_size=self.matrix_size,
                                           used_keypoints=self.used_keypoints,
                                           noise_frames=self.noise_frames,
                                           confidence_threshold=self.confidence_threshold)
                    video_data.load_xml_file(os.path.join(gesture_path, filename))
                    self._dataset[label].append(video_data)
                    self._flattened_matrices[label].append(video_data.get_flattened_matrix())

    def classify(self, runtime_matrix):
        """
        For now, tries to classify the given runtime matrix by performing the cross-correlation
        with all the flattened training matrices and taking the gesture for which the average
        peak-center distance is lowest.
        """

        lowest_distance = np.inf
        lowest_distance_label = None

        highest_magnitude = 0
        highest_magnitude_label = None

        for label in self._gesture_labels:
            flattened_matrices = self._flattened_matrices[label]
            peak_center_distances = []
            peak_magnitudes = []
            for other_matrix in flattened_matrices:
                corr = signal.fftconvolve(runtime_matrix, other_matrix[::-1, ::-1])

                # Find the peak of the cross-correlation.
                peak_index = np.unravel_index(np.argmax(corr, axis=None), corr.shape)

                # Compute the distance to the center.
                distance = np.sqrt(np.square(peak_index[0] - 62) + np.square(peak_index[1] - 62))
                peak_center_distances.append(distance)

                # Also consider the magnitude of the peak.
                magnitude = corr[peak_index]
                peak_magnitudes.append(magnitude)

            avg_distance = np.average(peak_center_distances)
            if avg_distance < lowest_distance:
                lowest_distance = avg_distance
                lowest_distance_label = label

            avg_magnitude = np.average(peak_magnitudes)
            if avg_magnitude > highest_magnitude:
                highest_magnitude = avg_magnitude
                highest_magnitude_label = label

        if lowest_distance_label == highest_magnitude_label and lowest_distance < 10 and highest_magnitude > 50:
            return lowest_distance_label, lowest_distance, highest_magnitude
        else:
            return None, lowest_distance, highest_magnitude
