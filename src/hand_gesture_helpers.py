from joblib import dump, load

import numpy as np
import cv2

from video_processing.keypoints import KEYPOINTS_DICT
from helpers import CLASS_DICT, CLASS_DICT_INVERSE

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

def get_wrist_shoulder_distance(keypoints):
    wrist_x = keypoints[KEYPOINTS_DICT["RWrist"]][0]
    wrist_y = keypoints[KEYPOINTS_DICT["RWrist"]][1]
    shoulder_x = keypoints[KEYPOINTS_DICT["RShoulder"]][0]
    shoulder_y = keypoints[KEYPOINTS_DICT["RShoulder"]][1]

    wrist_shoulder_distance = np.sqrt(np.square(wrist_x - shoulder_x) + np.square(wrist_y - shoulder_y))
    return wrist_shoulder_distance

def extract_hand_regions(frame, hand_rectangles):
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
        return CLASS_DICT["ZoomIn"]
    else:
        return CLASS_DICT["ZoomOut"]

def detect_hand_gesture_gradients(hand_region):
    # Simple rule: if horizontal gradients are greater than vertical gradients, it could be a palm.
    gradients_x, gradients_y = image_gradients(hand_region)
    if np.sum(np.abs(gradients_x)) > np.sum(np.abs(gradients_y)):
        return CLASS_DICT["ZoomIn"]
    else:
        return CLASS_DICT["ZoomOut"]

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

def save_matrix(event,x,y,flags,param):
    """
    Helper function used for data collection.
    """

    if event == cv2.EVENT_LBUTTONDOWN:
        video_data = deepcopy(param)
        video_data._frames = video_data._frames[-CONFIG["interpolation_frames"]:]
        video_data.save_to_xml("../crosscorr/%s.xml" % time.time())

# (From our process_videos.py)
def normalize_point(point_to_normalize, avg_dist, avg_point):
    nx = point_to_normalize[0] - avg_point[0]
    nx = nx / avg_dist
    ny = point_to_normalize[1] - avg_point[1]
    ny = ny / avg_dist

    return [nx,ny]