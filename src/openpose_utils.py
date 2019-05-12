import sys
import os
import argparse
import time
from sys import platform
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import cv2

from keypoints import KEYPOINTS_DICT

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/../openpose/lib/python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../openpose/lib/x64/Release;' +  dir_path + '/../openpose/bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('../../python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

def init_openpose(net_resolution="-1x368", hand_detection=False) -> op.WrapperPython:
    """
    Initializes OpenPose. For a list of parameters, refer to:
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/include/openpose/flags.hpp
    """

    # Set OpenPose options.
    params = dict()
    params["model_folder"] = "../openpose/models"
    params["model_pose"] = "BODY_25"
    params["net_resolution"] = net_resolution

    # NOTE: Activating hand detection is unnecessary now! Just keeping this code for reference.
    # if hand_detection:
    #     params["hand"] = 1
    #     params["hand_net_resolution"] = "16x16" # we're only using it for the hand rectangles, which work at any resolution

    # Start OpenPose.
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    return opWrapper

def process_image(imageToProcess: np.ndarray, opWrapper: op.WrapperPython) -> op.Datum:
    """
    Processes the image through OpenPose. Returns a Datum object:
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/include/openpose/core/datum.hpp
    """

    datum = op.Datum()
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])
    return datum

def get_distance(p1, p2):
    pixelX = p1[0] - p2[0]
    pixelY = p1[1] - p2[1]
    return np.sqrt(pixelX * pixelX + pixelY * pixelY)

def get_hand_rectangle_from_keypoints(wrist, elbow, shoulder) -> op.Rectangle:
    """
    Adapted from original CPP code:
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/hand/handDetector.cpp
    """

    ratio_wrist_elbow = 0.33 # Taken from openpose

    hand_rectangle = op.Rectangle()
    hand_rectangle.x = wrist[0] + ratio_wrist_elbow * (wrist[0] - elbow[0])
    hand_rectangle.y = wrist[1] + ratio_wrist_elbow * (wrist[1] - elbow[1])

    distance_wrist_elbow = get_distance(wrist,elbow)
    distance_elbow_shoulder = get_distance(elbow,shoulder)

    hand_rectangle.width = 1.5 * np.max([distance_wrist_elbow, 0.9 * distance_elbow_shoulder])
    hand_rectangle.height = hand_rectangle.width
    hand_rectangle.x -= hand_rectangle.width / 2.
    hand_rectangle.y -= hand_rectangle.width / 2.

    return hand_rectangle

def get_hand_rectangles_from_datum(datum: op.Datum) -> Optional[List[op.Rectangle]]:
    """
    Returns a list of two rectangles (left hand, right hand), or None (if no person was detected).
    If a rectangle is not detected, that value in the list will be None.
    NOTE: currently only computing the right hand rectangle.
    """

    keypoints = get_keypoints_from_datum(datum, ["RWrist", "RElbow", "RShoulder"])
    if keypoints is None:
        # Something went wrong: probably OpenPose did not detect any person. 
        return None

    hand_rectangles = [None, get_hand_rectangle_from_keypoints(keypoints[0], keypoints[1], keypoints[2])]
    return hand_rectangles

def get_keypoints_from_datum(datum: op.Datum, keypoints: List[str]) -> Optional[List[List[float]]]:
    """
    Returns the keypoints specified by name in the 'keypoints' argument.
    For a list of keypoints, look at KEYPOINTS_DICT in keypoints.py.
    Each returned keypoint is a list of three values: x, y and confidence.
    """

    poseKeypoints = datum.poseKeypoints
    if(poseKeypoints.size < 3):
        # Something went wrong: probably OpenPose did not detect any person.
        return None

    returnedKeypoints = []
    for k in keypoints:
        returnedKeypoints.append(poseKeypoints[0, KEYPOINTS_DICT[k]])

    return returnedKeypoints