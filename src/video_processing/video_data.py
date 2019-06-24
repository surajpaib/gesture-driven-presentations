# from __future__ import annotations
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import cv2

import sys
sys.path.append("./")

from video_processing.frame_data import FrameData
from video_processing.keypoints import KEYPOINTS_DICT
from config import CONFIG

class VideoData:
    def __init__(self, interpolations_frames=CONFIG["interpolation_frames"], matrix_size=CONFIG["matrix_size"], used_keypoints=CONFIG["used_keypoints"], noise_frames=CONFIG["noise_frames"], confidence_threshold=CONFIG["confidence_threshold"], matrix_vertical_crop=CONFIG["matrix_vertical_crop"]):
        self._frames = []
        # We have to save left out frames to save it back to an XML file
        self._pre_frames = [] # frames that are skipped before the start (noise frames)
        self._post_frames = [] # frames that are left out in the end (noise frames)
        self._label = ""
        self._framerate = 0

        self._matrix_list = []
        self._matrix_size = matrix_size
        self._matrix_vertical_crop = matrix_vertical_crop
        self._used_keypoints = [KEYPOINTS_DICT[k] for k in used_keypoints]
        
        self._last_keypoint_list = []
        while len(self._last_keypoint_list) < 18:
            self._last_keypoint_list.append([])

        self.noise_frames = noise_frames
        self.interpolation_frames = interpolations_frames
        self.confidence_threshold = confidence_threshold

    def get_keypoints_from_frame(self, frame_number):
        return (self._frames[frame_number]).keypoints

    def load_xml_file(self, xml_path):
        data_tree = ET.parse(xml_path)
        data_root = data_tree.getroot()
        label_node = data_root.find("Label")
        self._label = str(label_node.text)
        frame_rate_node = data_root.find("FPS")
        self._framerate = int(frame_rate_node.text)
        
        frame_list = []
        for frame_node in data_root.findall("Frame"):
            frame_data = FrameData()
            frame_data.load_from_xml(frame_node)
            frame_list.append(frame_data)


        for frame_data in frame_list[:self.noise_frames]:
            self._pre_frames.append(frame_data)
        for frame_data in frame_list[self.noise_frames : -self.noise_frames]:
            self.add_frame(frame_data)
        for frame_data in frame_list[-self.noise_frames:]:
            self._post_frames.append(frame_data)

    def add_frame(self, frame_data: FrameData):

        # Initialize the new interpolated matrix with the previous matrix.
        if len(self._matrix_list) > 0:
            matrix = self._matrix_list[-1]
        else:
            matrix = np.zeros((self._matrix_size - self._matrix_vertical_crop, self._matrix_size))
        self.prep_mat(matrix)

        # Iterate over the interesting keypoints.
        for k in self._used_keypoints:
            last_keypoints = self._last_keypoint_list[k][-(self.interpolation_frames - 1):]
            keypoint = frame_data.keypoints[k]

            # Apply the confidence threshold and either add the new point, or duplicate the last one.
            if keypoint[2] > self.confidence_threshold:
                key_x = int(keypoint[0] * self._matrix_size / 6 + self._matrix_size / 2)
                key_y = int(keypoint[1] * self._matrix_size / 6 + self._matrix_size / 6)
                if key_x >= self._matrix_size or key_y >= self._matrix_size - self._matrix_vertical_crop:
                    print("Error adding frame; key is to big: "+str(key_x) + " | " +str(key_y))
                    continue
                matrix[key_y, key_x] = 1
                last_keypoints.append([key_x, key_y])
            else:
                if len(last_keypoints) > 0:
                    last_keypoints.append(last_keypoints[-1])
                
            # Interpolate over the previous values.
            if len(last_keypoints) > 1:
                last_keypoints_x = [p[0] for p in last_keypoints]
                last_keypoints_y = [p[1] for p in last_keypoints]
                # f1 = interp1d(last_keypoints_x[-2:], last_keypoints_y[-2:], kind='linear')

                # # Find how many interpolation steps are needed from the last x to the previous x.
                # # Also determine a good step size (so that the values go from 0.75 to 1 linearly).
                # steps = abs(last_keypoints_x[-1] - last_keypoints_x[-2]) + 1
                # if steps == 1:
                #     continue
                # step_size = (1 / self.interpolation_frames) / (steps-1)     # (1 / self.interpolation_frames) gives 0.25 for 4 frames
                # step = 0
    
                # # Find the direction of the interpolation.
                # base = last_keypoints_x[-2]
                # direction = 1
                # if last_keypoints_x[-2] > last_keypoints_x[-1]:
                #     direction = -1

                # # Actually perform the interpolation.
                # for j in range(steps):
                #     x = base + direction * j
                #     matrix[int(f1(x)), x] = (1 - (1 / self.interpolation_frames)) + step * step_size    # (1 - (1 / self.interpolation_frames)) gives 0.75 for 4 frames
                #     step += 1
                x_A, x_B, y_A, y_B = last_keypoints_x[-2], last_keypoints_x[-1], last_keypoints_y[-2], last_keypoints_y[-1]
                d = np.sqrt(np.square(x_A - x_B) + np.square(y_A - y_B))
                # print("x_A: %d. x_B: %d. y_A: %d. y_B: %d. d: %.2f. int(d): %d" % (x_A, x_B, y_A, y_B, d, int(d)))
                if d > 1:
                    step_size = (1 / self.interpolation_frames) / int(d)
                    for i in range(int(d)):
                        x_i = int(x_A + i/d*(x_B-x_A))
                        y_i = int(y_A + i/d*(y_B-y_A))
                        matrix[y_i, x_i] = (1 - (1 / self.interpolation_frames)) + i * step_size
                        # print("\tx_i: %d. y_i: %d. Value: %.2f" % (x_i, y_i, matrix[y_i, x_i]))
                    matrix[y_B, x_B] = 1

            self._last_keypoint_list[k] = last_keypoints

        # Adding in the end to have the possibility of skipping invalid frames
        # Add frame data to the internal list.
        self._frames.append(frame_data)

        # Add the matrix to the list.
        self._matrix_list.append(matrix.copy())

    @property
    def frames(self):
        return self._frames

    @property
    def label(self):
        return self._label

    @property
    def fps(self):
        return self._framerate

    def get_matrices(self):
        return np.float32(np.array(self._matrix_list[self.interpolation_frames - 1:]))

    def get_newest_matrix(self):
        return np.float32(self._matrix_list[-1])

    def get_flattened_matrix(self):
        flattened_matrix = np.zeros((self._matrix_size - self._matrix_vertical_crop, self._matrix_size))
        for matrix in self._matrix_list:
            flattened_matrix += matrix
        return flattened_matrix

    def prep_mat(self, frame):
        for x in range(len(frame)):
            for y in range(len(frame[x])):
                if not frame[x][y] == 0:
                    frame[x][y] -= (1 / self.interpolation_frames)             # 0.25 because we are interpolating over 4 frames
                    if frame[x][y] < (1 / self.interpolation_frames):          
                        frame[x][y] = 0
    
    def save_to_xml(self,file_path:str = None):
        root = ET.Element('data')
        label_node = ET.SubElement(root,'Label')
        label_node.text = self._label
        fps_node = ET.SubElement(root,'FPS')
        fps_node.text = str(self._framerate)
        for frame in self._pre_frames:
            root.append(frame.create_xml_node())
        for frame in self._frames:
            root.append(frame.create_xml_node())
        for frame in self._post_frames:
            root.append(frame.create_xml_node())
        if file_path is not None:
            tree = ET.ElementTree(root)
            tree.write(file_path)
        return root
            

def get_final_matrix(interpolation_frames, filename):
    """
    Helper function to make code a bit cleaner.
    """

    data = VideoData(interpolation_frames)
    data.load_xml_file(filename)
    return data.get_flattened_matrix()

def get_matrix_list(interpolation_frames, filename):
    """
    Helper function to make code a bit cleaner.
    """

    data = VideoData(interpolation_frames)
    data.load_xml_file(filename)
    return data.get_matrices()

if __name__ == "__main__":
    data = VideoData(interpolation_frames=4, matrix_size=32, used_keypoints=["RWrist","LWrist"])
    data.load_xml_file("../process_results/test.xml")
    data.save_to_xml("../process_results/test_s.xml")
