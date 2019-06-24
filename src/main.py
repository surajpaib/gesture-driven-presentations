import time
import random
from typing import List, Tuple
from copy import deepcopy

import cv2
import imutils
import numpy as np

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise
from joblib import dump, load

from skimage import io
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color

import pyautogui
import pynput

from openpose_utils import *
from wrappers import PowerpointWrapper
from ml.autoencoder import *
from ml.classifier import Classifier
from ml.correlation_classifier import CorrelationClassifier
from openpose_utils import *
from video_processing.frame_data import FrameData
from video_processing.keypoints import get_all_keypoints_list, KEYPOINTS_DICT
from video_processing.video_data import VideoData

current_milli_time = lambda: int(round(time.time() * 1000))
timer_seconds = current_milli_time() / 1000

def blur_image(image):
    return cv2.blur(image, (3,3))

def sharpen_image(image):
    kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])
    return cv2.filter2D(image, -1, kernel_sharpening)

def image_gradients(image):
    img_grayscale = cv2.cvtColor(cv2.blur(image, (3,3)), cv2.COLOR_BGR2GRAY)
    gradients_x = cv2.Sobel(img_grayscale, cv2.CV_64F, 1, 0, ksize=5)
    gradients_y = cv2.Sobel(img_grayscale, cv2.CV_64F, 0, 1, ksize=5)
    return gradients_x, gradients_y


def extract_hand_regions(frame: np.ndarray, hand_rectangles: List[op.Rectangle]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crops out the two hand regions from the image, based on the given rectangles.
    """

    # Extract left hand rectangle if it was detected.
    if hand_rectangles[0] is not None:
        rect_x_1 = int(hand_rectangles[0].x)
        rect_y_1 = int(hand_rectangles[0].y)
        rect_x_2 = int(hand_rectangles[0].x + hand_rectangles[0].width)
        rect_y_2 = int(hand_rectangles[0].y + hand_rectangles[0].height)
        left_hand_region = frame.copy()[rect_y_1:rect_y_2, rect_x_1:rect_x_2]
    else:
        left_hand_region = None

    # Extract right hand rectangle if it was detected.
    if hand_rectangles[1] is not None:
        rect_x_1 = int(hand_rectangles[1].x)
        rect_y_1 = int(hand_rectangles[1].y)
        rect_x_2 = int(hand_rectangles[1].x + hand_rectangles[1].width)
        rect_y_2 = int(hand_rectangles[1].y + hand_rectangles[1].height)
        right_hand_region = frame.copy()[rect_y_1:rect_y_2, rect_x_1:rect_x_2]
    else:
        right_hand_region = None

    # Display the two rectangles.
    if left_hand_region is not None and left_hand_region.shape[0] > 0 and left_hand_region.shape[1] > 0:
        pass
        # cv2.imshow("Left hand region", left_hand_region)
    if right_hand_region is not None and right_hand_region.shape[0] > 0 and right_hand_region.shape[1] > 0:
        cv2.imshow("Right hand region", right_hand_region)

    # Return the two regions.
    return left_hand_region, right_hand_region

def basic_preprocessing_steps(img):
    new_img = cv2.blur(img, (5,5))
    return new_img

def condition_1(rgb_img, hsv_img, ycrcb_img):
    rgb = rgb_img
    hsv = hsv_img
    ycrcb = ycrcb_img
    return (hsv[:,:,0] >= 0) & (hsv[:,:,0] <= 25) & (hsv[:,:,1] >= 0.23 * 255) \
        & (hsv[:,:,1] <= 0.68 * 255) & (rgb[:,:,0] > 95) & (rgb[:,:,1] > 40) \
        & (rgb[:,:,2] > 20) & (rgb[:,:,0] > rgb[:,:,1]) & (rgb[:,:,0] > rgb[:,:,2]) \
        & (np.abs(rgb[:,:,0] - rgb[:,:,1]) > 15)

def condition_2(rgb_img, hsv_img, ycrcb_img):
    rgb = rgb_img
    hsv = hsv_img
    ycrcb = ycrcb_img
    return (rgb[:,:,0] > 95) & (rgb[:,:,1] > 40) & (rgb[:,:,2] > 20) \
        & (rgb[:,:,0] > rgb[:,:,1]) & (rgb[:,:,0] > rgb[:,:,2]) & (np.abs(rgb[:,:,0] - rgb[:,:,1]) > 15) \
        & (ycrcb[:,:,2] > 85) & (ycrcb[:,:,0] > 80) & (ycrcb[:,:,1] <= 1.5862*ycrcb[:,:,2]+20) \
        & (ycrcb[:,:,1] >= 0.3448*ycrcb[:,:,2]+76.2069) & (ycrcb[:,:,1] >= -4.5652*ycrcb[:,:,2]+234.5652) \
        & (ycrcb[:,:,1] <= -1.15*ycrcb[:,:,2]+301.75) & (ycrcb[:,:,1] <= -2.2857*ycrcb[:,:,2]+432.85)

def skin_segmentation_thresholds(img):
    """
    Based on https://arxiv.org/ftp/arxiv/papers/1708/1708.02694.pdf .
    Perform thresholding in RGB, HSV and YCrCb at the same time.
    
    NOTE: OpenCV color ranges are: H from 0-179, S and V from 0-255
    """
    
    img = basic_preprocessing_steps(img)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    
    mask = condition_1(rgb_img, hsv_img, ycrcb_img) | condition_2(rgb_img, hsv_img, ycrcb_img)
            
    return mask

def hand_segmentation(hand_image):
    # Perform thresholding segmentation and resize to a fixed size.
    mask = skin_segmentation_thresholds(hand_image).astype(np.uint8)
    mask = cv2.resize(mask, (150, 150), interpolation=cv2.INTER_NEAREST)
    
    # Apply some morphological operations.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask = cv2.dilate(mask, kernel, iterations = 1)
    
    # Extract contours and only keep the biggest contour.
    final_mask = np.zeros(mask.shape, np.uint8)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        max_contour = max(contours, key = cv2.contourArea)
        cv2.drawContours(final_mask, [max_contour], -1, (255,0,0), thickness=cv2.FILLED)
    
    return final_mask

def init_svm_classifier(classifier_path):
    """
    Loads a pretrained SVM hand gesture classifier
    """

    classifier = load(classifier_path) 
    return classifier

def detect_hand_gesture(classifier, hand_region):
    """
    Performs gesture detection on the right hand region and returns if
    the confidence is above a certain threshold.
    """

    CONFIDENCE_THRESHOLD = 0

    # Reshape and normalize values (classifier is trained with 0 and 1 values, but the obtained image
    # uses [0,255] range).
    classifier_input = (hand_region.reshape((1, -1)) / 255).astype(np.uint8)
    prediction = classifier.predict(classifier_input)
    if prediction == 0:
        return "PALM"
    else:
        return "FIST"

def detect_hand_gesture_gradients(hand_region):
    # Simple rule: if horizontal gradients are greater than vertical gradients, it could be a palm.
    gradients_x, gradients_y = image_gradients(hand_region)
    if np.sum(np.abs(gradients_x)) > np.sum(np.abs(gradients_y)):
        return "PALM"
    else:
        return "FIST"

def segment_hand_external_library(hand_region):
    import skin_detector

    img_msk = skin_detector.process(hand_region)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cv2.filter2D(img_msk, -1, disc, img_msk)

    final_mask = np.zeros(img_msk.shape, np.uint8)
    _, contours, _ = cv2.findContours(img_msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        max_contour = max(contours, key = cv2.contourArea)
        cv2.drawContours(final_mask, [max_contour], -1, (255,0,0), thickness=cv2.FILLED)

    hand_region_segmented = cv2.bitwise_and(hand_region, hand_region, mask=final_mask)

    return final_mask, hand_region_segmented

def segment_hand_histograms(hand_region):
    # Convert hand region to HSV.
    hand_region_resized = cv2.resize(hand_region, (150, 150), interpolation=cv2.INTER_NEAREST)
    hand_region_hsv = cv2.cvtColor(hand_region_resized, cv2.COLOR_BGR2HSV)

    # Take a small patch in the center of the hand region.
    patch_size = 25
    center_x = hand_region.shape[0] // 2
    center_y = hand_region.shape[1] // 2
    patch = hand_region_hsv[center_x-patch_size//2:center_x+patch_size//2, center_y-patch_size//2:center_y+patch_size//2]

    # Convert patch to HSV, compute histogram, normalize
    patch_hist = cv2.calcHist([patch], [0, 1], None, [75, 75], [0, 180, 0, 256])
    cv2.normalize(patch_hist, patch_hist, 0, 255, cv2.NORM_MINMAX)

    # Backproject histogram, binarize and mask the hand region.
    mask = cv2.calcBackProject([hand_region_hsv], [0, 1], patch_hist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cv2.filter2D(mask, -1, disc, mask)

    final_mask = np.zeros(mask.shape, np.uint8)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        max_contour = max(contours, key = cv2.contourArea)
        cv2.drawContours(final_mask, [max_contour], -1, (255,0,0), thickness=cv2.FILLED)

    masked_hand_region = cv2.bitwise_and(hand_region_resized, hand_region_resized, mask = final_mask)

    return final_mask, masked_hand_region

# (From our process_videos.py)
def normalize_point(point_to_normalize, avg_dist, avg_point):
    nx = point_to_normalize[0] - avg_point[0]
    nx = nx / avg_dist
    ny = point_to_normalize[1] - avg_point[1]
    ny = ny / avg_dist

    return [nx,ny]

def count_non_zero(flattened_data):
    count = 0
    flattened_data = np.round(flattened_data,1)
    for i in range(len(flattened_data)):
        if flattened_data[i] != 0:
            count += 1
    return count

def calculate_pixel_diff(orig_image, reconstructed_image):
    error = 0
    reconstructed_image = np.round(reconstructed_image[0, :], 1)
    for pos in range(len(orig_image)):
        orig_val = orig_image[pos]
        reconstructed_val = reconstructed_image[pos]
        if orig_val == reconstructed_val:
            continue
        elif (orig_val == 0) and not (reconstructed_val == 0):
            error += 1
        elif not (orig_val != 0 and reconstructed_val != 0):
            error += 1
    return error


def save_matrix(event,x,y,flags,param):
    """
    Helper function used for data collection.
    """

    if event == cv2.EVENT_LBUTTONDOWN:
        video_data = deepcopy(param)
        video_data._frames = video_data._frames[-CONFIG["interpolation_frames"]:]
        video_data.save_to_xml("../crosscorr/%s.xml" % time.time())

def get_cross_correlation_prediction(correlation_classifier, interpolated_frame):
    lowest_distance_label, lowest_distance, highest_magnitude_label, highest_magnitude = correlation_classifier.classify(interpolated_frame)
    prediction = CLASS_DICT["None"]
    if lowest_distance_label == highest_magnitude_label:
        if lowest_distance_label == "Reset7" and lowest_distance < 2 and highest_magnitude > 5:
            # print(lowest_distance_label, lowest_distance, highest_magnitude_label, highest_magnitude)
            prediction = CLASS_DICT["Reset"]
        elif lowest_distance_label == "RNext7Mirror" and highest_magnitude > 2.5:
            # print(lowest_distance_label, lowest_distance, highest_magnitude_label, highest_magnitude)
            prediction = CLASS_DICT["RNext"]
        elif lowest_distance_label == "LPrev7" and highest_magnitude > 3:
            # print(lowest_distance_label, lowest_distance, highest_magnitude_label, highest_magnitude)
            prediction = CLASS_DICT["LPrev"]
        elif lowest_distance_label == "StartStop7" and lowest_distance < 2.5 and highest_magnitude > 5:
            # print(lowest_distance_label, lowest_distance, highest_magnitude_label, highest_magnitude)
            prediction = CLASS_DICT["StartStop"]

    return prediction

def get_autoencoder_prediction(autoencoder, interpolated_frame):
    interpolated_frame_tensor = torch.from_numpy(interpolated_frame.reshape(1, -1))
    decoded = autoencoder.forward(interpolated_frame_tensor)
    cv2.imshow("Decoded frame", cv2.resize(decoded.data.numpy().reshape(interpolated_frame.shape), (320, 320)))
    reconstruction_error = calculate_pixel_diff(interpolated_frame.flatten(), decoded.data.numpy())
    non_zero_orig = count_non_zero((interpolated_frame.flatten()))

    if non_zero_orig > 8:
        if reconstruction_error > CONFIG["reconstruction_error_threshold"] or reconstruction_error < 10:
            print("BIG ERROR", reconstruction_error)
            return None
        else:
            print("SMALL ERROR", reconstruction_error)
            classes = classifier.forward(autoencoder.get_latent_space(interpolated_frame_tensor)).data.numpy()[0]
            gesture = np.argmax(classes)
            if classes[gesture] > 0.7:
                return gesture

# -------------------------------------------------------------------------------
# Main function
#-------------------------------------------------------------------------------

CLASS_DICT = {
    "None": -1,
    "LPrev": 0,
    "Reset": 1,
    "RNext": 2,
    "StartStop": 3
}

CLASS_DICT_INVERSE = {
    -1: "None",
    0: "LPrev",
    1: "Reset",
    2: "RNext",
    3: "StartStop"
}

if __name__ == "__main__":
    # Initialize OpenPose.
    opWrapper = init_openpose(net_resolution="176x-1", hand_detection=False)

    # Get the reference to the webcam.
    cv2.namedWindow('Video Feed')
    camera = cv2.VideoCapture(0)
    num_frames = 0

    presentation_opened = False
    wrapper = None

    # Load some relevant configuration values.
    interp_frames = CONFIG["interpolation_frames"]
    img_size = CONFIG["matrix_size"]
    used_keypoints = CONFIG["used_keypoints"]
    confidence_threshold = CONFIG["confidence_threshold"]
    matrix_vertical_crop = CONFIG["matrix_vertical_crop"]
    
    ### Initialisation of arm gesture classifier.
    if CONFIG["arm_gesture_classifier"] == "cross-correlation":
        video_data = VideoData(interp_frames, used_keypoints=used_keypoints, confidence_threshold=confidence_threshold,
                           matrix_size=img_size)

        # Load correlation classifier model.
        correlation_classifier = CorrelationClassifier()

    elif CONFIG["arm_gesture_classifier"] == "autoencoder":
        video_data = VideoData(interp_frames, used_keypoints=used_keypoints, confidence_threshold=confidence_threshold,
                           matrix_size=img_size)

        # Load autoencoder.
        autoencoder = Autoencoder(train_data_shape=(img_size * (img_size - matrix_vertical_crop)))
        autoencoder.load_state()

        # Load classifier.
        classifier = Classifier(4)
        classifier.load_state()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    elif CONFIG["arm_gesture_classifier"] == "heuristics":
        pass

    else:
        print("Invalid config value for arm_gesture_classifier!")
        exit(-1)
    ### END of arm gesture classifier initialisation.

    ### Initialisation of hand gesture classifier.
    keyboard = pynput.keyboard.Controller()
    mouse = pynput.mouse.Controller()
    hand_gesture_history = []
    if CONFIG["hand_gesture_classifier"] == "svm":
        hand_gesture_classifier = init_svm_classifier("models/hand_classifier_v1.joblib")

    elif CONFIG["hand_gesture_classifier"] == "gradients":
        pass

    else:
        print("Invalid config value for hand_gesture_classifier!")
        exit(-1)
    ### END of hand gesture classifier initialisation.

    # Keep looping, until interrupted by a Q keypress.
    time_window = 500 # ms
    gesture_history_size = 25
    time_window_beginning = current_milli_time()
    zoom_running = False
    frame_counter = 0
    while True:
        frame_counter += 1

        # Measure frame time.
        start_frame_time = current_milli_time()

        # Get the current frame, resize and flip it.
        (grabbed, frame) = camera.read()
        frame = imutils.resize(frame, width=700)
        (height, width) = frame.shape[:2]

        # Pass the frame through OpenPose.
        datum = process_image(frame, opWrapper)

        # Get some useful values from the Datum object.
        keypoints = get_keypoints_from_datum(datum, get_all_keypoints_list())
        hand_rectangles = get_hand_rectangles_from_datum(datum)

        # # Extract the hand rectangles, segment out the hand and perform some gesture detection (?).
        # if hand_rectangles:
        #     global left_hand_region, right_hand_region
        #     left_hand_region, right_hand_region = extract_hand_regions(frame, hand_rectangles)
        #     if right_hand_region is not None and right_hand_region.size != 0:

        #         # Perhaps use this wrist shoulder distance?
        #         wrist_x = keypoints[KEYPOINTS_DICT["RWrist"]][0]
        #         wrist_y = keypoints[KEYPOINTS_DICT["RWrist"]][1]
        #         shoulder_x = keypoints[KEYPOINTS_DICT["RShoulder"]][0]
        #         shoulder_y = keypoints[KEYPOINTS_DICT["RShoulder"]][1]
        #         cv2.circle(frame, (wrist_x, wrist_y), 10, (0, 0, 255), -1)
        #         cv2.circle(frame, (shoulder_x, shoulder_y), 10, (0, 0, 255), -1)
        #         wrist_shoulder_distance = np.sqrt(np.square(wrist_x - shoulder_x) + np.square(wrist_y - shoulder_y))

        #         if wrist_shoulder_distance < 50:
        #             # WITH SEGMENTATION
        #             right_hand_region_resized = cv2.resize(right_hand_region, (150, 150), interpolation=cv2.INTER_NEAREST)
        #             mask1, _ = segment_hand_external_library(right_hand_region_resized)
        #             mask2 = hand_segmentation(right_hand_region_resized)
        #             mask3, _ = segment_hand_histograms(right_hand_region_resized)
        #             final_mask = mask1 & mask2 & mask3
        #             cv2.imshow("Mask", mask1 & mask2 & mask3)

        #             right_hand_segmented = cv2.bitwise_and(right_hand_region_resized, right_hand_region_resized, mask = final_mask)
        #             hand_gesture = detect_hand_gesture(hand_gesture_classifier, final_mask)
        #             hand_gesture = detect_hand_gesture_gradients(right_hand_segmented)

        #             # NO SEGMENTATION:
        #             # right_hand_region_resized = cv2.resize(right_hand_region, (100, 100), interpolation=cv2.INTER_NEAREST)
        #             # right_hand_region_resized = cv2.resize(cv2.cvtColor(right_hand_region, cv2.COLOR_BGR2GRAY), (100, 100), interpolation=cv2.INTER_NEAREST)
        #             # right_hand_region_resized = cv2.resize(cv2.Canny(cv2.cvtColor(right_hand_region, cv2.COLOR_BGR2GRAY), 100, 200), (100, 100), interpolation=cv2.INTER_NEAREST)
        #             # hand_gesture = detect_hand_gesture(hand_gesture_classifier, right_hand_region_resized)
        #             print(hand_gesture)
                    
        #             # if hand_gesture == "PALM" and np.sum(final_mask) / np.size(final_mask) > 30:
        #             #     hand_gesture_history.append(hand_gesture)
        #             # else:
        #             #     hand_gesture_history.append("FIST")

        #     if keypoints[KEYPOINTS_DICT["RWrist"]][0] > 0 and keypoints[KEYPOINTS_DICT["RWrist"]][1] > 0:
                
        #         # Take the average point for the normalization as the right shoulder.
        #         avg_x = keypoints[KEYPOINTS_DICT["RShoulder"]][0]
        #         avg_y = keypoints[KEYPOINTS_DICT["RShoulder"]][1]
        #         # cv2.circle(frame, (int(avg_x), int(avg_y)), 10, (0,255,0), -1)
                
        #         # Compute average distance (1.5 shoulder lengths in this case)
        #         avg_dist = 1.5 * np.sqrt((keypoints[4][0]-keypoints[5][0])**2 + (keypoints[4][1]-keypoints[5][1])**2)

        #         # Normalize the wrist position according to these values.
        #         normalized_point = normalize_point([keypoints[0][0], keypoints[0][1]], avg_dist, [avg_x, avg_y])
        #         # print("Avg point: [%.3f, %.3f]. Avg distance: %.3f. Normalized point: [%.3f, %.3f]" % (avg_x, avg_y, avg_dist, normalized_point[0], normalized_point[1]))

        #         # Clamp to range [-1,1] then normalize to range [0,1]
        #         if normalized_point[0] < -1:
        #             normalized_point[0] = -1
        #         elif normalized_point[0] > 1:
        #             normalized_point[0] = 1
        #         if normalized_point[1] < -1:
        #             normalized_point[1] = -1
        #         elif normalized_point[1] > 1:
        #             normalized_point[1] = 1
        #         normalized_point[0] = (normalized_point[0] - (-1)) / 2
        #         normalized_point[1] = (normalized_point[1] - (-1)) / 2

        #         # Flip X-axis.
        #         normalized_point[0] = 1 - normalized_point[0]

        #         # VERSION 2
        #         # Now normalize to the range of the screen.
        #         x_normalized = normalized_point[0] * 1920
        #         y_normalized = normalized_point[1] * 1080

        #         # Filter to smooth the movement a little bit.
        #         N = 3
        #         smoothing_filter = np.ones(N) / N
        #         mouse_x_positions.append(x_normalized)
        #         mouse_y_positions.append(y_normalized)
        #         filtered_x = np.convolve(mouse_x_positions[-5:], smoothing_filter, mode='same')
        #         filtered_y = np.convolve(mouse_y_positions[-5:], smoothing_filter, mode='same')

        #         mouse.position = (filtered_x[-1], filtered_y[-1])

        # # Every time window, look through the history of hand gesture predictions.
        # # If there is a majority gesture, then perform it.
        # if current_milli_time() - time_window_beginning > time_window:
        #     hand_gesture_history = hand_gesture_history[-gesture_history_size:]
        #     samples = len(hand_gesture_history)
        #     # print("Palm count: %d. Fist count: %d. None count: %d." % (hand_gesture_history.count("PALM"), hand_gesture_history.count("FIST"), hand_gesture_history.count(None)))
        #     if hand_gesture_history.count("PALM") > 3/4*samples:
        #         # print("PALM DETECTED")
        #         if not zoom_running:
        #             keyboard.press(pynput.keyboard.Key.f12)
        #             keyboard.release(pynput.keyboard.Key.f12)
        #         if presentation_opened:
        #             presentation.start_zoom()
        #         zoom_running = True

        #     elif hand_gesture_history.count("FIST") > 3/4*samples:
        #         # print("FIST DETECTED")
        #         if zoom_running:
        #             keyboard.press(pynput.keyboard.Key.f12)
        #             keyboard.release(pynput.keyboard.Key.f12)
        #         if presentation_opened:
        #             presentation.stop_zoom()
        #         zoom_running = False

        #     time_window_beginning = current_milli_time()

        # If any person was detected on the screen, we can perform the classification.
        if keypoints is not None:
            # Build a FrameData object and add it to the VideoData to get the interpolation of the previous frames.
            frame_data = FrameData.from_keypoints(keypoints)
            video_data.add_frame(frame_data)

            # Now get the last matrix of VideoData (which should be the last 4 frames, interpolated)
            interpolated_frame = video_data.get_newest_matrix()

            # Dilate the interpolated frame (if enabled).
            if CONFIG["use_dilation"]:
                kernel_size = CONFIG["kernel_size"]
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                interpolated_frame = cv2.dilate(interpolated_frame, kernel, iterations=1)
            interpolated_frame_resized = cv2.resize(interpolated_frame, (interpolated_frame.shape[1] * 10, interpolated_frame.shape[0] * 10))
            cv2.imshow("Interpolated frame", interpolated_frame_resized)

            ### Arm gesture detection.
            if CONFIG["arm_gesture_classifier"] == "cross-correlation":
                prediction = get_cross_correlation_prediction(correlation_classifier, interpolated_frame)
            elif CONFIG["arm_gesture_classifier"] == "autoencoder":
                prediction = get_autoencoder_prediction(autoencoder, interpolated_frame)
            elif CONFIG["arm_gesture_classifier"] == "heuristics":
                pass
            print(CLASS_DICT_INVERSE[prediction])
            ### END arm gesture detection.

        # Stop measuring frame time and display FPS.
        end_frame_time = current_milli_time()
        frame_time = (end_frame_time - start_frame_time) / 1000  # seconds
        cv2.putText(frame, "FPS: %d" % int(1 / frame_time), (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the video frame.
        cv2.imshow("Video Feed", frame)

        # If the user pressed "Q", then quit.
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break
        

# Clean up.
camera.release()
cv2.destroyAllWindows()
if wrapper:
    wrapper.quit()
