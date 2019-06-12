import xml.etree.ElementTree as ET

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from frame_data import FrameData

class VideoData:
    def __init__(self):
        self._frames = []
        self._label = ""
        self._framerate = 0
    
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

    def generate_matrices(self):
        def sort_func(keypoint):
            return keypoint[0]
        matrix_list = []
        last_keypoint_list = []
        for i in range(len(self._frames)):
            frame = self._frames[i]
            matrix = np.zeros((64,64))
            if i != 0:
                matrix = matrix_list[i-1]
            self.prep_mat(matrix)
            for k in range(len(frame.keypoints[2:8])):
                if (len(last_keypoint_list)) <= k:
                    last_keypoint_list.append([])
                last_keypoints = last_keypoint_list[k]
                if(len(last_keypoints) > 3):
                    last_keypoints.pop(0)
                keypoint = (frame.keypoints[2:8])[k]
                if keypoint[2] > 0.5:
                    key_x = int(keypoint[0] * 16 + 32)
                    key_y = int(keypoint[1] * 16 + 8)
                    matrix[key_y,key_x] = 1
                    last_keypoints.append([key_x,key_y])
                else:
                    if len(last_keypoints) > 0:
                        last_keypoints.append(last_keypoints[-1])
                if len(last_keypoints) > 1:
                    #last_keypoints.sort(key=sort_func)
                    last_keypoints_x = [p[0] for p in last_keypoints]
                    last_keypoints_y = [p[1] for p in last_keypoints]
                    if k == 5:
                        print(last_keypoints)
                    f1 = interp1d(last_keypoints_x,last_keypoints_y,kind='linear')
                    steps =  max(last_keypoints_x) - min(last_keypoints_x) + 1
                    step_size = 0.75/steps
                    step = 0
                    for x in range(min(last_keypoints_x) + 1, max(last_keypoints_x)):
                        print(last_keypoints_x)
                        print(min(last_keypoints_x))
                        print(max(last_keypoints_x))
                        matrix[int(f1(x)),x] = 0.25 + step*step_size
                        step += 1
                last_keypoint_list[k] = last_keypoints
            matrix_list.append(matrix.copy())
        return matrix_list

    def prep_mat(self, frame):
        for x in range(len(frame)):
            for y in range(len(frame[x])):
                if not frame[x][y] == 0:
                    frame[x][y] -= 0.25
                    if frame[x][y] < 0.25:
                        frame[x][y] = 0

if __name__ == "__main__":
    data = VideoData()
    data.load_xml_file("../process_results/2019-05-02_11-57-40_7.xml")
    matrix_list = data.generate_matrices()
    for matrix in matrix_list[:9]:
        plt.figure()
        plt.imshow(matrix, cmap='gray')
        plt.show()
