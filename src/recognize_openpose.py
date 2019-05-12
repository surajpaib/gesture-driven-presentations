import cv2
import imutils
import numpy as np
import time
from typing import List, Tuple
from sklearn.metrics import pairwise

from skimage import io
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color

from wrappers import PowerpointWrapper, PresentationWrapper
from openpose_utils import *

current_milli_time = lambda: int(round(time.time() * 1000))
timer_seconds = current_milli_time() / 1000

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

#-------------------------------------------------------------------------------
# Main function
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    # Initialize OpenPose.
    opWrapper = init_openpose(net_resolution="176x-1", hand_detection=False)

    # Get the reference to the webcam.
    cv2.namedWindow('Video Feed')
    camera = cv2.VideoCapture(0)
    num_frames = 0

    presentation_opened = False
    wrapper = None

    # Used for basic wrist gesture detection.
    r_wrist_x_old = 0
    l_wrist_x_old = 0

    # Keep looping, until interrupted by a Q keypress.
    while(True):
        # Measure frame time.
        start_frame_time = current_milli_time()

        # Get the current frame, resize and flip it.
        (grabbed, frame) = camera.read()
        frame = imutils.resize(frame, width=700)
        frame_flipped = cv2.flip(frame, 1)
        (height, width) = frame.shape[:2]

        # Pass the frame through OpenPose.
        datum = process_image(frame, opWrapper)

        # Get some useful values from the Datum object.
        keypoints = get_keypoints_from_datum(datum, ["RWrist", "LWrist"])
        hand_rectangles = get_hand_rectangles_from_datum(datum)

        # Extract the hand rectangles, segment out the hand and perform some gesture detection (?).
        if hand_rectangles:
            global left_hand_region, right_hand_region
            left_hand_region, right_hand_region = extract_hand_regions(frame, hand_rectangles)
            if right_hand_region is not None and right_hand_region.size != 0:
                right_hand_segmented = hand_segmentation(right_hand_region)
                cv2.imshow("Right hand segmented", right_hand_segmented)

        # Perform gesture recognition on the arm keypoints.
        # if keypoints is not None:
        #     r_wrist = keypoints[0]
        #     l_wrist = keypoints[1]
        #     action = ""

        #     # Only track if both wrists are seen with a confidence of > 0.3.
        #     if (r_wrist[2] > 0.3) and (l_wrist[2] > 0.3):
        #         r_wrist_x = r_wrist[0]
        #         r_wrist_y = r_wrist[1]
        #         l_wrist_x = l_wrist[0]
        #         l_wrist_y = l_wrist[1]

        #         # Display circles on the wrists.
        #         cv2.circle(frame, (l_wrist_x, l_wrist_y), 10, (0,0,255), -1)
        #         cv2.circle(frame, (r_wrist_x, r_wrist_y), 10, (255,0,0), -1)

        #         # Left wrist: open/previous slide.
        #         if(l_wrist_x_old - l_wrist_x > 150):
        #             if not presentation_opened:
        #                 action = "OPEN"
        #                 wrapper = PowerpointWrapper()
        #                 presentation = wrapper.open_presentation("../MRP-6.pptx")
        #                 presentation.run_slideshow()
        #                 presentation_opened = True
        #             else:
        #                 action = "PREV"
        #                 presentation.previous_slide()
        #         l_wrist_x_old = l_wrist_x

        #         # Right wrist: open/next slide.
        #         if(r_wrist_x - r_wrist_x_old > 150):
        #             if not presentation_opened:
        #                 action = "OPEN"
        #                 wrapper = PowerpointWrapper()
        #                 presentation = wrapper.open_presentation("../MRP-6.pptx")
        #                 presentation.run_slideshow()
        #                 presentation_opened = True
        #             else:
        #                 presentation.next_slide()
        #                 action="NEXT"
        #         r_wrist_x_old = r_wrist_x

        #     # Show if any action has been done.
        #     cv2.putText(frame, str(action), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # Stop measuring frame time and display FPS.
        end_frame_time = current_milli_time()
        frame_time = (end_frame_time - start_frame_time) / 1000 # seconds
        cv2.putText(frame, "FPS: %d" % int(1/frame_time), (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # Display the video frame.
        cv2.imshow("Video Feed", frame)
        cv2.setMouseCallback("Video Feed", save_region)

        # If the user pressed "Q", then quit.
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break

# Clean up.
camera.release()
cv2.destroyAllWindows()
if wrapper:
    wrapper.quit()