# from __future__ import annotations
import xml.etree.ElementTree as ET

import numpy as np

import sys
sys.path.append("./")

from video_processing.process_videos_utils import normalize_point, create_xml_for_keypoint
from video_processing.keypoints import KEYPOINTS_DICT

class FrameData:
    def __init__(self):
        self._keypoints = []
        self._avg_point = [0,0]
        self._avg_dist = 0
    
    def load_from_xml(self, frame_node : ET.Element):
        avg_x_node = frame_node.find("Avg_x")
        self._avg_point[0] = int(avg_x_node.text)
        avg_y_node = frame_node.find("Avg_y")
        self._avg_point[1] = int(avg_y_node.text)
        avg_dist_node = frame_node.find("Avg_dist")
        self._avg_dist = float(avg_dist_node.text)
        for keypoint_node in frame_node.findall("Keypoint"):
            keypoint = [0,0,0]
            x_node = keypoint_node.find("X")
            keypoint[0] = float(x_node.text)
            y_node = keypoint_node.find("Y")
            keypoint[1] = float(y_node.text)
            confidence_node = keypoint_node.find("Confidence")
            keypoint[2] = float(confidence_node.text)
            self._keypoints.append(keypoint)

    def create_xml_node(self):
        frame_node = ET.Element('Frame')
        avg_x_node = ET.SubElement(frame_node,"Avg_x")
        avg_x_node.text = str(int(self.avg_point[0]))
        avg_y_node = ET.SubElement(frame_node,"Avg_y")
        avg_y_node.text = str(int(self.avg_point[1]))
        avg_dist_node = ET.SubElement(frame_node,"Avg_dist")
        avg_dist_node.text = str(int(self.avg_dist))
        for i in range(len(self.keypoints)):
            frame_node.append(create_xml_for_keypoint(i,self.keypoints[i],self.avg_dist,self.avg_point))
        return frame_node
    
    @staticmethod
    def from_keypoints(keypoints):
        """
        Expects all 18 keypoints that we use.
        """

        frame_data = FrameData()

        # Get the shoulder distance and the middle point between the shoulders.
        r_shoulder = keypoints[KEYPOINTS_DICT["RShoulder"]]
        l_shoulder = keypoints[KEYPOINTS_DICT["LShoulder"]]
        frame_data._avg_dist = np.sqrt((r_shoulder[0] - l_shoulder[0])**2 + (r_shoulder[1] - l_shoulder[1])**2)
        frame_data._avg_point = [(r_shoulder[0] + l_shoulder[0])/2, (r_shoulder[1] + l_shoulder[1])/2]

        # Normalize the rest of the keypoints.
        for k in keypoints:
            norm_point = normalize_point([k[0],k[1]], frame_data._avg_dist, frame_data._avg_point)
            frame_data._keypoints.append([norm_point[0], norm_point[1], k[2]])

        return frame_data

    @property
    def keypoints(self):
        return self._keypoints
    @property
    def avg_point(self):
        return self._avg_point
    @property
    def avg_dist(self):
        return self._avg_dist