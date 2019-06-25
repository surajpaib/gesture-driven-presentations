import time
import random
from copy import deepcopy

import cv2
import imutils
import numpy as np

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise

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
from hand_gesture_helpers import *
from stopwatch import Stopwatch
from helpers import CLASS_DICT, CLASS_DICT_INVERSE

current_milli_time = lambda: int(round(time.time() * 1000))
timer_seconds = current_milli_time() / 1000

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

    prediction = CLASS_DICT["None"]
    if non_zero_orig > 8:
        if reconstruction_error > CONFIG["reconstruction_error_threshold"] or reconstruction_error < 10:
            print("BIG ERROR", reconstruction_error)
        else:
            print("SMALL ERROR", reconstruction_error)
            classes = classifier.forward(autoencoder.get_latent_space(interpolated_frame_tensor)).data.numpy()[0]
            gesture = np.argmax(classes)
            if classes[gesture] > 0.7:
                prediction = gesture

    return prediction

# Store last "CONSIDERED_FRAMES" frames and boolean values for opening/closing the presentation.
# A bit dirty but easiest to do as global.
wrists_at_shoulders_booleans = []
keypoint_frame_lists = []
CONSIDERED_FRAMES = 10
START_STOP_REQUIRED_FRAMES = 5
ACCEPTABLE_DISTANCE_WRIST_SHIFT = 1.4 
ACCEPTABLE_DISTANCE_WRIST_RESET_SHIFT = 0.7
ACCEPTABLE_DISTANCE_WRIST_SHOULDER = 0.3
def get_heuristic_prediction(keypoints):
    global wrists_at_shoulders_booleans, keypoint_frame_lists

    # Extract the relevant keypoints.
    r_wrist = keypoints[KEYPOINTS_DICT["RWrist"]]
    l_wrist = keypoints[KEYPOINTS_DICT["LWrist"]]
    r_shoulder = keypoints[KEYPOINTS_DICT["RShoulder"]]
    l_shoulder = keypoints[KEYPOINTS_DICT["LShoulder"]]

    # Only track if all 4 keypoints are seen with a confidence of > 0.3.
    if (r_wrist[2] > 0.3) and (l_wrist[2] > 0.3) and (r_shoulder[2] > 0.3) and (l_shoulder[2] > 0.3):
        # Compute average point and average distance.
        avg_dist = np.sqrt((r_shoulder[0] - l_shoulder[0]) ** 2 + (r_shoulder[1] - l_shoulder[1]) ** 2)
        avg_point = [(r_shoulder[0] + l_shoulder[0]) / 2, (r_shoulder[1] + l_shoulder[1]) / 2]
        avg_point[0] = int(avg_point[0])
        avg_point[1] = int(avg_point[1])
    else:
        return CLASS_DICT["None"]

    # Normalize keypoints.
    r_wrist = normalize_point(r_wrist, avg_dist, avg_point)
    l_wrist = normalize_point(l_wrist, avg_dist, avg_point)
    r_shoulder = normalize_point(r_shoulder, avg_dist, avg_point)
    l_shoulder = normalize_point(l_shoulder, avg_dist, avg_point)

    # Build and maintain a list containing "CONSIDERED_FRAMES" (uniformly structured) dictionaries of normalized keypoints.
    selected_keypoints_dict = {"RWrist": r_wrist, "LWrist": l_wrist, "RShoulder": r_shoulder, "LShoulder": l_shoulder}
    keypoint_frame_lists.append(selected_keypoints_dict)
    if len(keypoint_frame_lists) > CONSIDERED_FRAMES:
        keypoint_frame_lists.pop(0)

    # Get current frame keypoint data.
    r_wrist_x = r_wrist[0]
    r_wrist_y = r_wrist[1]
    l_wrist_x = l_wrist[0]
    l_wrist_y = l_wrist[1]
    r_shoulder_x = r_shoulder[0]
    r_shoulder_y = r_shoulder[1]
    l_shoulder_x = l_shoulder[0]
    l_shoulder_y = l_shoulder[1]

    if len(keypoint_frame_lists) > 1:
        ### RNEXT, LPREV GESTURES ###
        # Get previous frame wrist x data.
        r_wrist_x_old = keypoint_frame_lists[-2]["RWrist"][0]
        l_wrist_x_old = keypoint_frame_lists[-2]["LWrist"][0]

        # No absolute value applied, since then also backward movements will be caught
        # (e.g. right arm going back from the left to the right).
        if r_wrist_x - r_wrist_x_old > ACCEPTABLE_DISTANCE_WRIST_SHIFT:
            return CLASS_DICT["RNext"]
        if l_wrist_x_old - l_wrist_x > ACCEPTABLE_DISTANCE_WRIST_SHIFT:
            return CLASS_DICT["LPrev"]
        ### END RNEXT, LPREV GESTURES ###

        ### RESET GESTURE ###
        # [Measured experimentally] At least 6 frames are required for for the reset gesture with current distance values.
        if len(keypoint_frame_lists) > 5:
            r_wrist_x_fifth_old = keypoint_frame_lists[-5]["RWrist"][0]
            l_wrist_x_fifth_old = keypoint_frame_lists[-5]["LWrist"][0]

            r_wrist_x_sixth_old = keypoint_frame_lists[-6]["RWrist"][0]
            l_wrist_x_sixth_old = keypoint_frame_lists[-6]["LWrist"][0]

            # Condition 1 and 2, first movements of the right and left arm respectively. They need to be between 
            # 2 relevant thresholds so as not to overlap with RNEXT and LPREV gestures.
            cond1 = ACCEPTABLE_DISTANCE_WRIST_SHIFT > r_wrist_x_fifth_old - r_wrist_x_sixth_old > ACCEPTABLE_DISTANCE_WRIST_RESET_SHIFT 
            cond2 = ACCEPTABLE_DISTANCE_WRIST_SHIFT > r_wrist_x - r_wrist_x_old > ACCEPTABLE_DISTANCE_WRIST_RESET_SHIFT

            # Condition 3 and 4, third movements of the right and left arm. Analogous constraints as for condition 1 and 2.
            cond3 = ACCEPTABLE_DISTANCE_WRIST_SHIFT > l_wrist_x_sixth_old - l_wrist_x_fifth_old > ACCEPTABLE_DISTANCE_WRIST_RESET_SHIFT
            cond4 = ACCEPTABLE_DISTANCE_WRIST_SHIFT > l_wrist_x_old - l_wrist_x > ACCEPTABLE_DISTANCE_WRIST_RESET_SHIFT

            if cond1 and cond2 and cond3 and cond4:
                return CLASS_DICT["Reset"]
        ### END RESET GESTURE ###

        ### START-STOP GESTURE ###
        # For appropriate working of the program, no other PPT presentation should be open beforehand.
        right_wrist_at_right_shoulder = (np.absolute(np.absolute(r_wrist_x) - np.absolute(r_shoulder_x)) < ACCEPTABLE_DISTANCE_WRIST_SHOULDER) \
            and (np.absolute(np.absolute(r_wrist_y) - np.absolute(r_shoulder_y)) < ACCEPTABLE_DISTANCE_WRIST_SHOULDER)
        left_wrist_at_left_shoulder = (np.absolute(np.absolute(l_wrist_x) - np.absolute(l_shoulder_x)) < ACCEPTABLE_DISTANCE_WRIST_SHOULDER) \
            and (np.absolute(np.absolute(l_wrist_y) - np.absolute(l_shoulder_y)) < ACCEPTABLE_DISTANCE_WRIST_SHOULDER)

        if right_wrist_at_right_shoulder and left_wrist_at_left_shoulder:
            wrists_at_shoulders_booleans.append(True)
            print("Accumulating for START/STOP")
        else:
            wrists_at_shoulders_booleans.append(False)

        if len(wrists_at_shoulders_booleans) > CONSIDERED_FRAMES:
            wrists_at_shoulders_booleans.pop(0)

        if sum(wrists_at_shoulders_booleans[-START_STOP_REQUIRED_FRAMES:]) == START_STOP_REQUIRED_FRAMES:
            wrists_at_shoulders_booleans = []
            return CLASS_DICT["StartStop"]
        ### END START STOP GESTURE ###

    return CLASS_DICT["None"]

def segment_hand_region(right_hand_region):
    # Resize to a standard size.
    right_hand_region_resized = cv2.resize(right_hand_region, (150, 150), interpolation=cv2.INTER_NEAREST)
    
    # "Average" the result of the three segmentation methods.
    mask1, _ = segment_hand_external_library(right_hand_region_resized)
    mask2 = hand_segmentation(right_hand_region_resized)
    mask3, _ = segment_hand_histograms(right_hand_region_resized)
    final_mask = mask1 & mask2 & mask3
    cv2.imshow("Mask", final_mask)

    right_hand_segmented = cv2.bitwise_and(right_hand_region_resized, right_hand_region_resized, mask = final_mask)
    return final_mask, right_hand_segmented

def get_svm_prediction(hand_gesture_classifier, right_hand_region):
    final_mask, right_hand_segmented = segment_hand_region(right_hand_region)
    hand_gesture = detect_hand_gesture(hand_gesture_classifier, final_mask)
    
    if hand_gesture == CLASS_DICT["ZoomIn"] and np.sum(final_mask) / np.size(final_mask) > 30:
        return CLASS_DICT["ZoomIn"]
    else:
        return CLASS_DICT["ZoomOut"]

def get_gradients_prediction(right_hand_region):
    final_mask, right_hand_segmented = segment_hand_region(right_hand_region)
    hand_gesture = detect_hand_gesture_gradients(right_hand_segmented)

    if hand_gesture == CLASS_DICT["ZoomIn"] and np.sum(final_mask) / np.size(final_mask) > 30:
        return CLASS_DICT["ZoomIn"]
    else:
        return CLASS_DICT["ZoomOut"]

def get_mouse_position(keypoints, mouse_x_positions, mouse_y_positions):
    # Compute average point and average distance for normalization.
    avg_x = keypoints[KEYPOINTS_DICT["RShoulder"]][0]
    avg_y = keypoints[KEYPOINTS_DICT["RShoulder"]][1]
    avg_dist = 1.5 * np.sqrt((keypoints[4][0]-keypoints[5][0])**2 + (keypoints[4][1]-keypoints[5][1])**2)

    # Normalize the wrist position.
    normalized_point = normalize_point([keypoints[KEYPOINTS_DICT["RWrist"]][0], keypoints[KEYPOINTS_DICT["RWrist"]][1]], avg_dist, [avg_x, avg_y])

    # Clamp to range [-1,1] then normalize to range [0,1].
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

    # Flip X-axis.
    normalized_point[0] = 1 - normalized_point[0]

    # Now normalize to the range of the screen.
    x_normalized = normalized_point[0] * 1920
    y_normalized = normalized_point[1] * 1080

    # Filter to smooth the movement a little bit.
    N = CONFIG["hand_motion_smoothing_frames"]
    if N > 0:
        smoothing_filter = np.ones(N) / N
        mouse_x_positions.append(x_normalized)
        mouse_y_positions.append(y_normalized)
        filtered_x = np.convolve(mouse_x_positions[-5:], smoothing_filter, mode='same')[-1]
        filtered_y = np.convolve(mouse_y_positions[-5:], smoothing_filter, mode='same')[-1]
    else:
        filtered_x = mouse_x_position[-1]
        filtered_y = mouse_y_position[-1]

    return filtered_x, filtered_y

def get_majority_hand_gesture(hand_gesture_history):
    samples = len(hand_gesture_history)
    
    if hand_gesture_history.count("PALM") > 3/4*samples:
        return CLASS_DICT["ZoomIn"]
    elif hand_gesture_history.count("FIST") > 3/4*samples:
        return CLASS_DICT["ZoomOut"]

def display_keypoint(frame, keypoints, keypoint_name):
    k = keypoints[KEYPOINTS_DICT[keypoint_name]]
    cv2.circle(frame, (int(k[0]), int(k[1])), 10, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), -1)


# -------------------------------------------------------------------------------
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
    predicting = False
    wrapper = None
    pause_stopwatch = Stopwatch()

    # Load some relevant configuration values.
    interp_frames = CONFIG["interpolation_frames"]
    img_size = CONFIG["matrix_size"]
    used_keypoints = CONFIG["used_keypoints"]
    confidence_threshold = CONFIG["confidence_threshold"]
    matrix_vertical_crop = CONFIG["matrix_vertical_crop"]
    
    ### Initialisation of arm gesture classifier.
    video_data = VideoData(interp_frames, used_keypoints=used_keypoints, confidence_threshold=confidence_threshold,
                        matrix_size=img_size)
    arm_gesture_text_timer = Stopwatch()
    arm_gesture_displaying = ""
    if CONFIG["arm_gesture_classifier"] == "cross-correlation":
        correlation_classifier = CorrelationClassifier()

    elif CONFIG["arm_gesture_classifier"] == "autoencoder":
        # Load autoencoder.
        autoencoder = Autoencoder(train_data_shape=(img_size * (img_size - matrix_vertical_crop)))
        autoencoder.load_state()

        # Load classifier.
        classifier = Classifier(4)
        classifier.load_state()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    elif CONFIG["arm_gesture_classifier"] == "heuristics":
        # Moved as global things for now. TODO: refactor a little bit and add stuff to config maybe?
        pass

    else:
        print("Invalid config value for arm_gesture_classifier!")
        exit(-1)
    ### END of arm gesture classifier initialisation.

    ### Initialisation of hand gesture classifier.
    if CONFIG["hand_gestures_enabled"] is True:
        keyboard = pynput.keyboard.Controller()
        mouse = pynput.mouse.Controller()
        hand_gesture_history = []
        mouse_x_positions = []
        mouse_y_positions = []
        hand_gesture_stopwatch = Stopwatch()

        if CONFIG["hand_gesture_classifier"] == "svm":
            hand_gesture_classifier = init_svm_classifier(CONFIG["hand_gesture_classifier_path"])

        elif CONFIG["hand_gesture_classifier"] == "gradients":
            # Nothing to initialise.
            pass

        else:
            print("Invalid config value for hand_gesture_classifier!")
            exit(-1)
    ### END of hand gesture classifier initialisation.

    # Keep looping, until interrupted by a Q keypress.
    while True:

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

        # If any person was detected on the screen, we can perform the classification.
        if keypoints is not None:
            # Build a FrameData object and add it to the VideoData to get the interpolation of the previous frames.
            frame_data = FrameData.from_keypoints(keypoints)
            video_data.add_frame(frame_data)

            # Now get the last matrix of VideoData (which should be the last 4 frames, interpolated)
            interpolated_frame = video_data.get_newest_matrix()

            # Display the wrist keypoints.
            display_keypoint(frame, keypoints, "RWrist")
            display_keypoint(frame, keypoints, "LWrist")
            display_keypoint(frame, keypoints, "RShoulder")
            display_keypoint(frame, keypoints, "LShoulder")

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
                prediction = get_heuristic_prediction(keypoints)
            
            if prediction != CLASS_DICT["None"]:
                print("Arm: ", CLASS_DICT_INVERSE[prediction])

            # Perform the StartStop action separately.
            if prediction == CLASS_DICT["StartStop"]:
                arm_gesture_text_timer.start()
                arm_gesture_displaying = CLASS_DICT_INVERSE[prediction]
                if not presentation_opened:
                    wrapper = PowerpointWrapper()
                    presentation = wrapper.open_presentation(CONFIG["presentation_path"])
                    presentation.run_slideshow()
                    presentation_opened = True
                    predicting = True
                    pause_stopwatch.start()
                else:
                    predicting = not predicting

            # Only perform the other actions if the system is currently
            # predicting and with at least 1 second pause between gestures.
            if prediction != CLASS_DICT["None"] and predicting and pause_stopwatch.time_elapsed > 1:
                arm_gesture_text_timer.start()
                arm_gesture_displaying = CLASS_DICT_INVERSE[prediction]
                if prediction == CLASS_DICT["RNext"]:
                    presentation.next_slide()
                elif prediction == CLASS_DICT["LPrev"]:
                    presentation.previous_slide()
                if prediction == CLASS_DICT["Reset"]:
                    pass

                pause_stopwatch.start()
            ### END arm gesture detection.

            ### Hand gesture detection.
            if CONFIG["hand_gestures_enabled"] is True and hand_rectangles:
                # First, see what prediction is obtained for this frame.
                if hand_rectangles:
                    left_hand_region, right_hand_region = extract_hand_regions(frame, hand_rectangles)
                    if right_hand_region is not None and right_hand_region.size != 0:
                        wrist_shoulder_distance = get_wrist_shoulder_distance(keypoints)

                        # Only predict if the wrist is close to the shoulder.
                        if wrist_shoulder_distance < 50:
                            if CONFIG["hand_gesture_classifier"] == "svm":
                                prediction = get_svm_prediction(hand_gesture_classifier, right_hand_region)
                            elif CONFIG["hand_gesture_classifier"] == "gradients":
                                prediction = get_gradients_prediction(right_hand_region)

                        # Next, look in the history of predictions for the majority prediction.
                        hand_gesture_history.append(prediction)
                        hand_gesture_history = hand_gesture_history[-CONFIG["hand_gesture_window_history_size"]:]
                        print("Palm count: %d. Fist count: %d." % (hand_gesture_history.count(CLASS_DICT["ZoomIn"]), hand_gesture_history.count(CLASS_DICT["ZoomOut"])))
                        if hand_gesture_stopwatch.time_elapsed < CONFIG["hand_gesture_time_window_s"]:
                            majority_gesture = get_majority_hand_gesture(hand_gesture_history)
                            
                            # If needed, perform an action on the presentation.
                            print("Hand: ", CLASS_DICT_INVERSE[majority_gesture])

                            # Restart the timer.
                            hand_gesture_stopwatch.start()
                        
                    # If needed, move the mouse cursor.
                    if presentation.zoom:
                        new_mouse_position = get_mouse_position(keypoints, mouse_x_positions, mouse_y_positions)
                        mouse.position = (new_mouse_position[0], new_mouse_position[1])
            ### END hand gesture detection.

        # Stop measuring frame time and display FPS.
        end_frame_time = current_milli_time()
        frame_time = (end_frame_time - start_frame_time) / 1000  # seconds
        cv2.putText(frame, "FPS: %d" % int(1 / frame_time), (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the last detected gesture for 1 second.
        prediction_window = np.zeros((100, 500, 3), np.uint8)
        if arm_gesture_displaying != "":
            if arm_gesture_text_timer.time_elapsed < 1:
                cv2.putText(prediction_window, arm_gesture_displaying, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
            else:
                arm_gesture_displaying = ""
        if arm_gesture_displaying == "" and not predicting:
            cv2.putText(prediction_window, "Stopped", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
        cv2.imshow("Prediction", prediction_window)

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
