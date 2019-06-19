import sys
import os
from sys import platform
import argparse
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

import xml.etree.ElementTree as ET

from pathlib import Path

from openpose_utils import *
from keypoints import KEYPOINTS_DICT
from process_videos_utils import normalize_point

openpose_initialized = False
opWrapper = None

def create_xml_for_keypoint(id,keypoint, avg_dist, avg_point):
    keypoint_node = ET.Element('Keypoint')
    id_node = ET.SubElement(keypoint_node,'ID')
    id_node.text = str(id)
    norm_point = normalize_point([keypoint[0],keypoint[1]],avg_dist,avg_point)
    x_node = ET.SubElement(keypoint_node,'X')
    x_node.text = str(norm_point[0])
    y_node = ET.SubElement(keypoint_node,'Y')
    y_node.text = str(norm_point[1])
    conf_node = ET.SubElement(keypoint_node,'Confidence')
    conf_node.text = str(keypoint[2])
    return keypoint_node



def get_frame_list_of_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_list = []

    while(cap.isOpened()):
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_list.append(frame)

        #cv2.imshow('frame',gray)

    print("Finished video")
    cap.release()
    return (frame_list, frame_rate)

def process_video(video_path, label, output_path,output_path_7, update=False):
    if(output_path.exists() and not update):
        return
    frame_list,frame_rate = get_frame_list_of_video(video_path)
    root = ET.Element('data')
    root_7 = ET.Element('data')
    label_node = ET.SubElement(root,'Label')
    label_node.text = str(label)
    label_node_7 = ET.SubElement(root_7,'Label')
    label_node_7.text = str(label)
    frame_rate_node = ET.SubElement(root,"FPS")
    frame_rate_node.text = str(frame_rate)
    frame_rate_node_7 = ET.SubElement(root_7,"FPS")
    frame_rate_node_7.text = str(7)
    tree = ET.ElementTree(root)
    tree_7 = ET.ElementTree(root_7)
    frame_rate_reduce_count = 0
    for input_frame in frame_list:
        if(frame_rate_reduce_count == 30):
            frame_rate_reduce_count = 0
        frame_node = ET.Element('Frame')
        # Pass the frame through OpenPose.
        datum = process_image(input_frame, opWrapper)
        # Get some useful values from the Datum object.
        keypoints = get_all_keypoints_from_datum(datum)
        if keypoints is None:
            continue

        r_shoulder = keypoints[KEYPOINTS_DICT["RShoulder"]]
        l_shoulder = keypoints[KEYPOINTS_DICT["LShoulder"]]
        avg_dist = np.sqrt((r_shoulder[0] - l_shoulder[0])**2 + (r_shoulder[1] - l_shoulder[1])**2)
        avg_point = [(r_shoulder[0] + l_shoulder[0])/2, (r_shoulder[1] + l_shoulder[1])/2]
        avg_point[0] = int(avg_point[0])
        avg_point[1] = int(avg_point[1])

        avg_x_node = ET.SubElement(frame_node,"Avg_x")
        avg_x_node.text = str(avg_point[0])
        avg_y_node = ET.SubElement(frame_node,"Avg_y")
        avg_y_node.text = str(avg_point[1])
        avg_dist_node = ET.SubElement(frame_node,"Avg_dist")
        avg_dist_node.text = str(avg_dist)

        ''' Normalization test
        r_wrist = keypoints[KEYPOINTS_DICT["RWrist"]]
        print(r_wrist)
        r_wrist_x = r_wrist[0] - avg_point[0]
        r_wrist_x = r_wrist_x / avg_dist
        r_wrist_y = r_wrist[1] - avg_point[1]
        r_wrist_y = r_wrist_y / avg_dist

        r_wrist_reverse_x = r_wrist_x * avg_dist
        r_wrist_reverse_x = r_wrist_x + avg_point[0]
        r_wrist_reverse_y = r_wrist_y * avg_dist
        r_wrist_reverse_y = r_wrist_y + avg_point[1]
        


        show_frame = datum.cvInputData
        cv2.circle(show_frame, (avg_point[0], avg_point[1]), 10, (0, 0, 255), -1)
        cv2.circle(show_frame, (int(r_wrist_reverse_x), int(r_wrist_reverse_y)), 10, (0, 0, 255), -1)
        cv2.circle(show_frame, (int(r_wrist[0]), int(r_wrist[1])), 10, (255, 0, 0), -1)
        cv2.imshow('frame',show_frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        '''
        for i in range(len(keypoints)):
            frame_node.append(create_xml_for_keypoint(i,keypoints[i],avg_dist,avg_point))
        root.append(frame_node)
        if(frame_rate_reduce_count % 4 == 0):
            root_7.append(frame_node)
        frame_rate_reduce_count += 1
    tree.write(output_path)
    tree_7.write(output_path_7)

    print("Finished")
if __name__ == "__main__":
    video_dir = Path('C:\\Users\\AGANDO\\Videos\\videos\\start_stop')
    result_dir = video_dir.parent / "Processing_Results_Start_Stop"
    result_dir_7 = video_dir.parent / "Processing_Results_Start_Stop_7"
    opWrapper = init_openpose(hand_detection=False)
    count = 0
    for video in video_dir.iterdir():
        if(video.is_file()):
            output_file_path = result_dir / (video.stem + ".xml")
            output_file_path_7 = result_dir_7 / (video.stem + "_7.xml")
            print(str(output_file_path))
            process_video(str(video),"Start_Stop",output_file_path,output_file_path_7,update=True)
            count = count + 1
    print("Finished")