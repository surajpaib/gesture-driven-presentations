import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from frame_data import FrameData


class VideoData:
    def __init__(self, interpolations_frames, noise_frames=2):
        self._frames = []
        self._label = ""
        self._framerate = 0
        self.noise_frames = noise_frames
        self.interpolation_frames = interpolations_frames

    def get_keypoints_from_frame(self, frame_number):
        return (self._frames[frame_number]).keypoints

    def load_xml_file(self, xml_path):
        data_tree = ET.parse(xml_path)
        data_root = data_tree.getroot()
        label_node = data_root.find("Label")
        self._label = str(label_node.text)
        frame_rate_node = data_root.find("FPS")
        self._framerate = int(frame_rate_node.text)
        for frame_node in data_root.findall("Frame"):
            frame_data = FrameData()
            frame_data.load_from_xml(frame_node)
            self._frames.append(frame_data)

    @property
    def frames(self):
        return self._frames

    @property
    def label(self):
        return self._label

    @property
    def fps(self):
        return self._framerate

    def generate_matrices(self, matrix_size):
        def sort_func(keypoint):
            return keypoint[0]

        MATRIX_SIZE = matrix_size
        matrix_list = []
        last_keypoint_list = []
        for i in range(self.noise_frames, len(self._frames) - self.noise_frames - 1):
            frame = self._frames[i]
            matrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE))

            if i != self.noise_frames:
                matrix = matrix_list[i - self.noise_frames - 1]

            self.prep_mat(matrix)
            for k in range(len(frame.keypoints[2:8])):
                if (len(last_keypoint_list)) <= k:
                    last_keypoint_list.append([])

                # Take the past 3 values for this current keypoint.
                last_keypoints = last_keypoint_list[k]
                if len(last_keypoints) > 3:     # we interpolate based on the 4 frames, so only keep the last 3 values.
                    last_keypoints.pop(0)

                # If we have a high enough confidence, add the current value. Else, duplicate the previous one.
                keypoint = (frame.keypoints[2:8])[k]
                if keypoint[2] > 0.5:           # apply a confidence threshold.
                    key_x = int(
                        keypoint[0] * MATRIX_SIZE / 4 + MATRIX_SIZE / 2)
                    key_y = int(
                        keypoint[1] * MATRIX_SIZE / 4 + MATRIX_SIZE / 8)
                    matrix[key_y, key_x] = 1
                    last_keypoints.append([key_x, key_y])
                else:                           # if the confidence threshold isn't met, duplicate the previous value
                    if len(last_keypoints) > 0:
                        last_keypoints.append(last_keypoints[-1])
                
                # Interpolate over the previous values.
                if len(last_keypoints) > 1:
                    last_keypoints_x = [p[0] for p in last_keypoints]
                    last_keypoints_y = [p[1] for p in last_keypoints]
                    f1 = interp1d(last_keypoints_x[-2:],
                                  last_keypoints_y[-2:], kind='linear')
                    steps = abs(last_keypoints_x[-1] - last_keypoints_x[-2]) + 1
                    if steps == 1:
                        continue
                    step_size = 0.25 / (steps-1)
                    step = 0

                    base = last_keypoints_x[-2]
                    direction = 1
                    if last_keypoints_x[-2] > last_keypoints_x[-1]:
                        direction = -1

                    for j in range(steps):
                        x = base + direction * j
                        matrix[int(f1(x)), x] = 0.75 + step * step_size
                        step += 1

                last_keypoint_list[k] = last_keypoints
            matrix_list.append(matrix.copy())

        # -2 because first index is 0
        return np.float32(np.array(matrix_list[self.interpolation_frames - 2:]))

    def prep_mat(self, frame):
        for x in range(len(frame)):
            for y in range(len(frame[x])):
                if not frame[x][y] == 0:
                    frame[x][y] -= 0.25             # 0.25 because we are interpolating over 4 frames
                    if frame[x][y] < 0.25:
                        frame[x][y] = 0


if __name__ == "__main__":
    data = VideoData(4)
    data.load_xml_file("../process_results/2019-05-02_11-57-40_7.xml")
    matrix_list = data.generate_matrices(64)
    for matrix in matrix_list[:9]:
        plt.figure()
        plt.imshow(matrix, cmap='gray')
        plt.show()
