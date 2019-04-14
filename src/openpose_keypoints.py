# From Python
# It requires OpenCV installed for Python
import sys
import os
from sys import platform
import argparse
import time
import cv2
import matplotlib.pyplot as plt
# Keep following on top to import openpose
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

def init_openpose():
    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="../examples/images/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
    args = parser.parse_known_args()
    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../openpose/models"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    return opWrapper

# Creates datum from image and returns right wrist and left wrist with confidence 
# The return value is [-1,-1,0] if something did not work
def getKeypointsFromImage(imageToProcess,opWrapper):
    datum = op.Datum()
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])

    return getKeypointsFromDatum(datum)

# returns right wrist and left wrist with confidence values from datum
# The return value is [-1,-1,0] if something did not work
def getKeypointsFromDatum(datum):
    #print("Body keypoints: \n" + str(datum.poseKeypoints))
    poseKeypoints = datum.poseKeypoints
    if(poseKeypoints.size < 3):
        return [-1,-1,0],[-1,-1,0] 
    numberPeople = poseKeypoints.shape[0]
    numberBodyParts = poseKeypoints.shape[1]
    # wrists are keypoint 4 and 7
    '''
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
    Result for BODY_25 (25 body parts consisting of COCO + foot)
         {0,  "Nose"},
         {1,  "Neck"},
         {2,  "RShoulder"},
         {3,  "RElbow"},
         {4,  "RWrist"},
         {5,  "LShoulder"},
         {6,  "LElbow"},
         {7,  "LWrist"},
         {8,  "MidHip"},
         {9,  "RHip"},
         {10, "RKnee"},
         {11, "RAnkle"},
         {12, "LHip"},
         {13, "LKnee"},
         {14, "LAnkle"},
         {15, "REye"},
         {16, "LEye"},
         {17, "REar"},
         {18, "LEar"},
         {19, "LBigToe"},
         {20, "LSmallToe"},
         {21, "LHeel"},
         {22, "RBigToe"},
         {23, "RSmallToe"},
         {24, "RHeel"},
         {25, "Background"}
    '''
    rWrist = poseKeypoints[0,4]
    lWrist = poseKeypoints[0,7]
    return (rWrist,lWrist)


if __name__ == "__main__":
    # Flags and init of openpose
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="./examples/images/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
    args = parser.parse_known_args()
    opWrapper = init_openpose()

    # Read frames on directory
    imagePaths = op.get_images_on_directory(args[0].image_dir)
    start = time.time()

    # Process images
    for imagePath in imagePaths:
        imageToProcess = cv2.imread(imagePath)
        r_wrist, l_wrist = getKeypointsFromImage(imageToProcess,opWrapper)
        r_Wrist_x = r_wrist[0]
        r_Wrist_y = r_wrist[1]
        l_Wrist_x = l_wrist[0]
        l_Wrist_y = l_wrist[1]
        image_show = imageToProcess
        cv2.circle(image_show, (r_Wrist_x, r_Wrist_y), 10, (255,0,0), -1)
        cv2.circle(image_show, (l_Wrist_x, l_Wrist_y), 10, (0,0,255), -1)
        # Convert image from BGR to RGB because opencv stores it in BGR
        image_show = cv2.cvtColor(image_show,cv2.COLOR_BGR2RGB)
        # Show image with matplotlib because it is more convenient to use
        plt.figure()
        plt.imshow(image_show)
    end = time.time()
    # Show all images
    plt.show()
    print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")