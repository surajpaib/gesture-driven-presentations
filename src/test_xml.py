import xml.etree.ElementTree as ET
import os

from video_processing.keypoints import KEYPOINTS_DICT

source_dir = "../dataset/LPrev7"
dest_dir = "../dataset/RNext7Mirror"
to_remove = "RWrist"

# source_dir = "../dataset/RNext7"
# dest_dir = "../dataset/RNext7Edited"
# to_remove = "LWrist"

for filename in os.listdir(source_dir):
    if filename.endswith(".xml"):
        data_tree = ET.parse(os.path.join(source_dir, filename))
        data_root = data_tree.getroot()
        for frame_node in data_root.findall("Frame"):
            for keypoint_node in frame_node.findall("Keypoint"):
                # keypoint_id = keypoint_node.find("ID").text
                # if int(keypoint_id) == KEYPOINTS_DICT[to_remove]:
                #     keypoint_confidence = keypoint_node.find("Confidence")
                #     keypoint_confidence.text = "0"

                keypoint_x = float(keypoint_node.find("X").text)
                keypoint_node.find("X").text = str(-keypoint_x)

        data_tree.write(os.path.join(dest_dir, filename))