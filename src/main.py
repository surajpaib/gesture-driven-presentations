import time
import random
from typing import List, Tuple

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
from video_processing.frame_data import FrameData
from video_processing.keypoints import get_all_keypoints_list
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

def save_region(event,x,y,flags,param):
    """
    Helper function used for data collection.
    """

    global left_hand_region, right_hand_region
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("SAVING REGION")
        # if left_hand_region is not None:
        #     cv2.imwrite("data/left_hand_fist/%d.png" % (int(current_milli_time()) // 5), left_hand_region)
        # if left_hand_region is not None:
        #     cv2.imwrite("data/left_hand_palm/%d.png" % (int(current_milli_time()) // 5), left_hand_region)
        # if right_hand_region is not None:
        #     cv2.imwrite("data/right_hand_palm/%d.png" % (int(current_milli_time()) // 5), right_hand_region)
        # if right_hand_region is not None:
            # cv2.imwrite("data/right_hand_fist/%d.png" % (int(current_milli_time()) // 5), right_hand_region)

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

    CONFIDENCE_THRESHOLD = 0.6

    # Reshape and normalize values (classifier is trained with 0 and 1 values, but the obtained image
    # uses [0,255] range).
    classifier_input = (hand_region.reshape((1, -1)) / 255).astype(np.uint8)
    probabilities = classifier.predict_proba(classifier_input)
    # print(probabilities)
    if probabilities[0][0] > CONFIDENCE_THRESHOLD:
        return "PALM"
    elif probabilities[0][1] > CONFIDENCE_THRESHOLD:
        return "FIST"
    else:
        return None

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
    # cv2.imshow("mask", img_msk)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cv2.filter2D(img_msk, -1, disc, img_msk)
    # img_msk = cv2.dilate(img_msk, disc, iterations = 1)

    final_mask = np.zeros(img_msk.shape, np.uint8)
    _, contours, _ = cv2.findContours(img_msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        max_contour = max(contours, key = cv2.contourArea)
        cv2.drawContours(final_mask, [max_contour], -1, (255,0,0), thickness=cv2.FILLED)

    hand_region_segmented = cv2.bitwise_and(hand_region, hand_region, mask=final_mask)
    # cv2.imshow("Segmented", hand_region_segmented)

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
    # mask = cv2.dilate(mask, disc, iterations = 1)

    final_mask = np.zeros(mask.shape, np.uint8)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        max_contour = max(contours, key = cv2.contourArea)
        cv2.drawContours(final_mask, [max_contour], -1, (255,0,0), thickness=cv2.FILLED)

    # cv2.imshow("Final", final_mask)

    # ret, thresh = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)

    # cv2.imshow("Thresh", thresh)

    # masked_hand_region = cv2.bitwise_and(hand_region, hand_region, mask = thresh)
    masked_hand_region = cv2.bitwise_and(hand_region_resized, hand_region_resized, mask = final_mask)

    # cv2.imshow("Masked", masked_hand_region)

    return final_mask, masked_hand_region

# (From our process_videos.py)
def normalize_point(point_to_normalize, avg_dist, avg_point):
    nx = point_to_normalize[0] - avg_point[0]
    nx = nx / avg_dist
    ny = point_to_normalize[1] - avg_point[1]
    ny = ny / avg_dist

    return [nx,ny]


# -------------------------------------------------------------------------------
# Main function
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    # Initialize OpenPose.
    opWrapper = init_openpose(net_resolution="176x-1", hand_detection=False)

    keyboard = pynput.keyboard.Controller()
    mouse = pynput.mouse.Controller()

    # Load the SVM hand gesture classifier.
    hand_gesture_classifier = init_svm_classifier("models/hand_classifier_v1.joblib")
    # hand_gesture_classifier = init_svm_classifier("models/hand_classifier_v2_logistic.joblib")
    # hand_gesture_classifier = init_svm_classifier("models/hand_classifier_v3_linear.joblib")

    hand_gesture_history = []

    # Get the reference to the webcam.
    cv2.namedWindow('Video Feed')
    camera = cv2.VideoCapture(0)
    num_frames = 0

    presentation_opened = False
    wrapper = None

    # Used for basic wrist gesture detection.
    r_wrist_x_old = 0
    l_wrist_x_old = 0

    # Used to smooth the cursor movement a little bit.
    mouse_x_positions = []
    mouse_y_positions = []

    # Used to automatically interpolate previous frames.
    interp_frames = CONFIG["interpolation_frames"]
    img_size = CONFIG["matrix_size"]
    used_keypoints = CONFIG["used_keypoints"]
    confidence_threshold = CONFIG["confidence_threshold"]
    video_data = VideoData(interp_frames, used_keypoints=used_keypoints, confidence_threshold=confidence_threshold,
                           matrix_size=img_size)

    # Correlation-based "classifier" initialisation.
    correlation_classifier = CorrelationClassifier()

    # Load the models
    matrix_vertical_crop = CONFIG["matrix_vertical_crop"]
    autoencoder = Autoencoder(train_data_shape=(img_size * (img_size - matrix_vertical_crop)))
    classifier = Classifier(4)

    # autoencoder.train()
    autoencoder.load_state()

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
        # frame_flipped = cv2.flip(frame, 1)
        (height, width) = frame.shape[:2]

        # Pass the frame through OpenPose.
        datum = process_image(frame, opWrapper)

        # Get some useful values from the Datum object.
        keypoints = get_keypoints_from_datum(datum, ["RWrist", "LWrist", "RElbow", "LElbow", "RShoulder", "LShoulder"])
        hand_rectangles = get_hand_rectangles_from_datum(datum)

        # Extract the hand rectangles, segment out the hand and perform some gesture detection (?).
        if hand_rectangles:
            global left_hand_region, right_hand_region
            left_hand_region, right_hand_region = extract_hand_regions(frame, hand_rectangles)
            if right_hand_region is not None and right_hand_region.size != 0:
                # right_hand_segmented = hand_segmentation(right_hand_region)
                # cv2.imshow("Right hand segmented", right_hand_segmented)

                # hand_gesture = detect_hand_gesture(hand_gesture_classifier, right_hand_segmented)
                # segmented_hand = segment_hand_histograms(right_hand_region)
                
                # VARIANT 1: could be decent.
                # segmented_hand = segment_hand_external_library(right_hand_region)
                # cv2.imshow("Segmented hand", segmented_hand)
                # hand_gesture = detect_hand_gesture_gradients(segmented_hand)
                # hand_gesture_history.append(hand_gesture)

                # VARIANT 2:
                # right_hand_region_resized = cv2.resize(right_hand_region, (150, 150), interpolation=cv2.INTER_NEAREST)
                # right_hand_mask = hand_segmentation(right_hand_region)
                # cv2.imshow("Mask", right_hand_mask)

                # disk_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
                # eroded_mask = cv2.erode(right_hand_mask, disk_el)
                # cv2.imshow("Eroded mask", eroded_mask)

                # # right_hand_region_resized[eroded_mask == 0] = 0
                # blurred = cv2.blur(right_hand_region_resized, (5,5))
                # # blurred_mask = cv2.blur(eroded_mask, (5,5))
                # # result = (blurred / blurred_mask[:, :, None]).astype(np.float32)
                # result = right_hand_region_resized + ((blurred - right_hand_region_resized) & eroded_mask[:, :, None])
                # cv2.imshow("Blurred inside of hand", result)

                # # right_hand_segmented = cv2.bitwise_and(result, result, mask = right_hand_mask)
                # hand_gesture = detect_hand_gesture_gradients(result)
                # hand_gesture_history.append(hand_gesture)

                # VARIANT 3:
                # right_hand_segmented = segment_hand_histograms(right_hand_region)
                # cv2.imshow("Segmented hand", right_hand_segmented)
                # hand_gesture = detect_hand_gesture_gradients(right_hand_segmented)
                # hand_gesture_history.append(hand_gesture)
                
                # ONE MORE IDEA: try all segmentation methods
                right_hand_region_resized = cv2.resize(right_hand_region, (150, 150), interpolation=cv2.INTER_NEAREST)
                mask1, _ = segment_hand_external_library(right_hand_region_resized)
                mask2 = hand_segmentation(right_hand_region_resized)
                mask3, _ = segment_hand_histograms(right_hand_region_resized)
                final_mask = mask1 & mask2 & mask3
                cv2.imshow("Mask", mask1 & mask2 & mask3)

                right_hand_segmented = cv2.bitwise_and(right_hand_region_resized, right_hand_region_resized, mask = final_mask)
                hand_gesture = detect_hand_gesture_gradients(right_hand_segmented)
                
                if hand_gesture == "PALM" and np.sum(final_mask) / np.size(final_mask) > 30:
                    hand_gesture_history.append(hand_gesture)
                else:
                    hand_gesture_history.append("FIST")

            if keypoints[0][0] > 0 and keypoints[0][1] > 0:
                
                # Compute average point (middle of two shoulders)
                # avg_x = (keypoints[4][0] + keypoints[5][0]) / 2
                # avg_y = (keypoints[4][1] + keypoints[5][1]) / 2
                # cv2.circle(frame, (int(avg_x), int(avg_y)), 10, (0,255,0), -1)
                # Can take avg point as right shoulder.
                avg_x = keypoints[4][0]
                avg_y = keypoints[4][1]
                cv2.circle(frame, (int(avg_x), int(avg_y)), 10, (0,255,0), -1)
                
                # Compute average distance (1.5 shoulder lengths in this case)
                avg_dist = 1.5 * np.sqrt((keypoints[4][0]-keypoints[5][0])**2 + (keypoints[4][1]-keypoints[5][1])**2)

                # Normalize the wrist position according to these values.
                normalized_point = normalize_point([keypoints[0][0], keypoints[0][1]], avg_dist, [avg_x, avg_y])
                # print("Avg point: [%.3f, %.3f]. Avg distance: %.3f. Normalized point: [%.3f, %.3f]" % (avg_x, avg_y, avg_dist, normalized_point[0], normalized_point[1]))

                # Clamp to range [-1,1] then normalize to range [0,1]
                if normalized_point[0] < -1:
                    normalized_point[0] = -1
                elif normalized_point[0] > 1:
                    normalized_point[0] = 1
                if normalized_point[1] < -1:
                    normalized_point[1] = -1
                elif normalized_point[1] > 1:
                    normalized_point[1] = 1
                normalized_point[0] = (normalized_point[0] - (-1)) / 2
                normalized_point[1] = (normalized_point[1] - (-1)) / 2

                # print(pyautogui.size(), keypoints[0][0], keypoints[0][1])

                # x_normalized = pyautogui.size()[0] - ((keypoints[0][0] - 250) / 130 * pyautogui.size()[0]) # invert X
                # y_normalized = (keypoints[0][1] - 140) / 180 * pyautogui.size()[1]

                # Flip X-axis.
                normalized_point[0] = 1 - normalized_point[0]

                # VERSION 1
                # Filter to smooth the movement a little bit.
                # N = 5
                # smoothing_filter = np.ones(N) / N
                # mouse_x_positions.append(normalized_point[0])
                # mouse_y_positions.append(normalized_point[1])
                # filtered_x = np.convolve(mouse_x_positions[-5:], smoothing_filter, mode='same')
                # filtered_y = np.convolve(mouse_y_positions[-5:], smoothing_filter, mode='same')

                # x_normalized = filtered_x[0] * 1920
                # y_normalized = filtered_y[1] * 1080

                # mouse.position = (x_normalized, y_normalized)

                # VERSION 2
                # Now normalize to the range of the screen.
                x_normalized = normalized_point[0] * 1920
                y_normalized = normalized_point[1] * 1080

                # Filter to smooth the movement a little bit.
                N = 3
                smoothing_filter = np.ones(N) / N
                mouse_x_positions.append(x_normalized)
                mouse_y_positions.append(y_normalized)
                filtered_x = np.convolve(mouse_x_positions[-5:], smoothing_filter, mode='same')
                filtered_y = np.convolve(mouse_y_positions[-5:], smoothing_filter, mode='same')

                mouse.position = (filtered_x[-1], filtered_y[-1])

                # pyautogui.moveTo(x_normalized, y_normalized)

        # Every time window, look through the history of hand gesture predictions.
        # If there is a majority gesture, then perform it.
        if current_milli_time() - time_window_beginning > time_window:
            hand_gesture_history = hand_gesture_history[-gesture_history_size:]
            samples = len(hand_gesture_history)
            print("Palm count: %d. Fist count: %d. None count: %d." % (hand_gesture_history.count("PALM"), hand_gesture_history.count("FIST"), hand_gesture_history.count(None)))
            if hand_gesture_history.count("PALM") > 3/4*samples:
                print("PALM DETECTED")
                if not zoom_running:
                    keyboard.press(pynput.keyboard.Key.f12)
                    keyboard.release(pynput.keyboard.Key.f12)
                if presentation_opened:
                    presentation.start_zoom()
                zoom_running = True

            elif hand_gesture_history.count("FIST") > 3/4*samples:
                print("FIST DETECTED")
                if zoom_running:
                    keyboard.press(pynput.keyboard.Key.f12)
                    keyboard.release(pynput.keyboard.Key.f12)
                if presentation_opened:
                    presentation.stop_zoom()
                zoom_running = False

            time_window_beginning = current_milli_time()
        keypoints = get_keypoints_from_datum(datum, get_all_keypoints_list())
        hand_rectangles = get_hand_rectangles_from_datum(datum)

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
                interpolated_frame_resized = cv2.resize(interpolated_frame, (320, 320))
            else:
                interpolated_frame_resized = cv2.resize(interpolated_frame, (320, 320))
            cv2.imshow("Interpolated frame", interpolated_frame_resized)

            # Get prediction from cross-correlation classifier?
            label, lowest_distance, highest_magnitude = correlation_classifier.classify(interpolated_frame)
            print(label, lowest_distance, highest_magnitude)

        # Perform gesture recognition on the arm keypoints.
        if keypoints is not None:
            r_wrist = keypoints[KEYPOINTS_DICT["RWrist"]]
            l_wrist = keypoints[KEYPOINTS_DICT["LWrist"]]
            action = ""

            # Only track if both wrists are seen with a confidence of > 0.3.
            if (r_wrist[2] > 0.3) and (l_wrist[2] > 0.3):
                r_wrist_x = r_wrist[0]
                r_wrist_y = r_wrist[1]
                l_wrist_x = l_wrist[0]
                l_wrist_y = l_wrist[1]

                # Display circles on the wrists.
                cv2.circle(frame, (l_wrist_x, l_wrist_y), 10, (0, 0, 255), -1)
                cv2.circle(frame, (r_wrist_x, r_wrist_y), 10, (255, 0, 0), -1)

                # Left wrist: open/previous slide.
                if l_wrist_x_old - l_wrist_x > 150:
                    if not presentation_opened:
                        action = "OPEN"
                        wrapper = PowerpointWrapper()
                        presentation = wrapper.open_presentation(CONFIG["presentation_path"])
                        presentation.run_slideshow()
                        presentation_opened = True
                    else:
                        action = "PREV"
                        presentation.previous_slide()
                l_wrist_x_old = l_wrist_x

                # Right wrist: open/next slide.
                if r_wrist_x - r_wrist_x_old > 150:
                    if not presentation_opened:
                        action = "OPEN"
                        wrapper = PowerpointWrapper()
                        presentation = wrapper.open_presentation(CONFIG["presentation_path"])
                        presentation.run_slideshow()
                        presentation_opened = True
                    else:
                        presentation.next_slide()
                        action = "NEXT"
                r_wrist_x_old = r_wrist_x

            # Show if any action has been done.
            cv2.putText(frame, str(action), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Stop measuring frame time and display FPS.
        end_frame_time = current_milli_time()
        frame_time = (end_frame_time - start_frame_time) / 1000  # seconds
        cv2.putText(frame, "FPS: %d" % int(1 / frame_time), (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the video frame.
        cv2.imshow("Video Feed", frame)
        cv2.setMouseCallback("Video Feed", save_region)

        # If the user pressed "Q", then quit.
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break
        if keypress == ord("n"):
            cv2.imwrite("wrist_picture.png", frame)
            cv2.imwrite("hand_region.png", right_hand_region)

        # Call the autoencoder
        interpolated_frame_tensor = torch.from_numpy(interpolated_frame.reshape(1, -1))
        decoded = autoencoder.forward(interpolated_frame_tensor)
        cv2.imshow("Decoded frame", cv2.resize(decoded.data.numpy().reshape(interpolated_frame.shape), (320, 320)))
        mse_error = (np.square(decoded.data.numpy().reshape(interpolated_frame.shape) - interpolated_frame)).mean(
            axis=None)
        if mse_error > CONFIG["reconstruction_error_threshold"]:
            print("BIG ERROR", mse_error)
        else:
            print("SMALL ERROR", mse_error)
            gesture = classifier.forward(autoencoder.get_latent_space(interpolated_frame_tensor))
            print("GESTURE DONE", gesture)

# Clean up.
camera.release()
cv2.destroyAllWindows()
if wrapper:
    wrapper.quit()
