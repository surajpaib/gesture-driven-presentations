import sys
import os
from sys import platform
from typing import List, Optional

import numpy as np

from video_processing.keypoints import KEYPOINTS_DICT
from config import CONFIG

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/../openpose/lib/python/openpose/Release')
        os.environ['PATH'] = os.environ[
                                 'PATH'] + ';' + dir_path + '/../openpose/lib/x64/Release;' + dir_path + '/../openpose/bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('../../python')
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e


def init_openpose(net_resolution=CONFIG["net_resolution"], hand_detection=False) -> op.WrapperPython:
    """
    Initializes OpenPose. For a list of parameters, refer to:
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/include/openpose/flags.hpp
    """

    # Set OpenPose options.
    params = dict()
    params["model_folder"] = CONFIG["model_folder"]
    params["model_pose"] = CONFIG["model_pose"]
    params["net_resolution"] = net_resolution
    if hand_detection:
        params["hand"] = 1
        params[
            "hand_net_resolution"] = "16x16"  # we're only using it for the hand rectangles, which work at any resolution

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


def get_hand_rectangles_from_datum(datum: op.Datum) -> Optional[List[op.Rectangle]]:
    """
    Returns a list of two rectangles (left hand, right hand), or None (if no person was detected).
    NOTE: the rectangles might be (0,0,0,0) if the corresponding hand was not detected!
    """

    if len(datum.handRectangles) > 0:
        return datum.handRectangles[0]
    else:
        # No person detected.
        return None

def get_all_keypoints_from_datum(datum: op.Datum)  -> Optional[List[List[float]]]:
    """
    Returns all keypoints
    Each returned keypoint is a list of three values: x, y and confidence.
    """

    poseKeypoints = datum.poseKeypoints
    if (poseKeypoints.size < 3):
        print("something wrong")
        # Something went wrong: probably OpenPose did not detect any person.
        return None

    return poseKeypoints[0]

def get_keypoints_from_datum(datum: op.Datum, keypoints: List[str]) -> Optional[List[List[float]]]:
    """
    Returns the keypoints specified by name in the 'keypoints' argument.
    For a list of keypoints, look at KEYPOINTS_DICT in keypoints.py.
    Each returned keypoint is a list of three values: x, y and confidence.
    """

    poseKeypoints = datum.poseKeypoints
    if (poseKeypoints.size < 3):
        # Something went wrong: probably OpenPose did not detect any person.
        return None

    returnedKeypoints = []
    for k in keypoints:
        returnedKeypoints.append(poseKeypoints[0, KEYPOINTS_DICT[k]])

    return returnedKeypoints
