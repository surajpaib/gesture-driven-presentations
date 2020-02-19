import time

import imutils

from openpose_utils import *
from video_processing.frame_data import FrameData
from video_processing.video_data import VideoData
from wrappers import PowerpointWrapper
from video_processing.process_videos_utils import *

import cv2

current_milli_time = lambda: int(round(time.time() * 1000))

# Minimum: max value of the following:
# * 6 frames, due to the reset gesture,
# * START_STOP_REQUIRED_FRAMES.
CONSIDERED_FRAMES = 10

# Maximum "CONSIDERED_FRAMES" frames.
START_STOP_REQUIRED_FRAMES = 5

# Normalized, acceptable distances.
ACCEPTABLE_DISTANCE_WRIST_SHIFT = 1.4 
ACCEPTABLE_DISTANCE_WRIST_RESET_SHIFT = 0.7
ACCEPTABLE_DISTANCE_WRIST_SHOULDER = 0.3


# -------------------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize OpenPose.
    opWrapper = init_openpose(net_resolution="176x-1", hand_detection=False)

    # Get the reference to the webcam.
    camera = cv2.VideoCapture(0)
    num_frames = 0

    presentation_opened = False
    wrapper = None

    # This flag is used to prevent from closing the presentation just after opening, 
    # due to potentially too long START/STOP gesture realization
    other_gesture_after_opening_performed = False

    # Store last "CONSIDERED_FRAMES" frames and boolean values for opening/closing the presentation.
    keypoint_frame_lists = []
    wrists_at_shoulders_booleans = []

    # Keep looping, until interrupted by a Q keypress.
    while True:
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
        keypoints = get_keypoints_from_datum(datum,
                                             ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow",
                                              "LWrist"])
        hand_rectangles = get_hand_rectangles_from_datum(datum)

        # Perform gesture recognition on the arm keypoints.
        if keypoints is not None:
            r_wrist = keypoints[4]
            l_wrist = keypoints[7]
            r_shoulder = keypoints[2]
            l_shoulder = keypoints[5]

            # Only track if all 4 keypoints are seen with a confidence of > 0.3.
            if (r_wrist[2] > 0.3) and (l_wrist[2] > 0.3) and (r_shoulder[2] > 0.3) and (l_shoulder[2] > 0.3):

                # Compute average point and average distance.
                avg_dist = np.sqrt((r_shoulder[0] - l_shoulder[0]) ** 2 + (r_shoulder[1] - l_shoulder[1]) ** 2)
                avg_point = [(r_shoulder[0] + l_shoulder[0]) / 2, (r_shoulder[1] + l_shoulder[1]) / 2]
                avg_point[0] = int(avg_point[0])
                avg_point[1] = int(avg_point[1])

                # Display circles on the wrists and shoulders. Use non-normalized points.
                cv2.circle(frame, (l_wrist[0], l_wrist[1]), 10, (0, 0, 255), -1)
                cv2.circle(frame, (r_wrist[0], r_wrist[1]), 10, (255, 0, 0), -1)
                cv2.circle(frame, (r_shoulder[0], r_shoulder[1]), 10, (255, 255, 0), -1)
                cv2.circle(frame, (l_shoulder[0], l_shoulder[1]), 10, (255, 0, 255), -1)

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

                if len(keypoint_frame_lists) > 1 and presentation_opened:
                    ### RNEXT, LPREV GESTURES ###

                    # Get previous frame wrist x data.
                    r_wrist_x_old = keypoint_frame_lists[-2]["RWrist"][0]
                    l_wrist_x_old = keypoint_frame_lists[-2]["LWrist"][0]

                    # No absolute value applied, since then also backward movements will be caught
                    # (e.g. right arm going back from the left to the right).
                    if r_wrist_x - r_wrist_x_old > ACCEPTABLE_DISTANCE_WRIST_SHIFT:
                        print("RIGHT ARM SHIFTED")
                        other_gesture_after_opening_performed = True
                        presentation.next_slide()

                    if l_wrist_x_old - l_wrist_x > ACCEPTABLE_DISTANCE_WRIST_SHIFT:
                        other_gesture_after_opening_performed = True
                        print("LEFT ARM SHIFTED")
                        presentation.previous_slide()
                    ### END RNEXT, LPREV GESTURES ###


                    # [Measured experimentally] At least 6 frames are required for for the reset gesture with current distance values.
                    if len(keypoint_frame_lists) > 5:
                        #### RESET GESTURE ###
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
                            other_gesture_after_opening_performed = True
                            print("RESET")
                            presentation.close_slideshow()
                            presentation.run_slideshow()
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
                    
                    wrists_at_shoulders_booleans[-1] = False

                    if not presentation_opened:
                        wrapper = PowerpointWrapper()
                        presentation = wrapper.open_presentation("../MRP-6.pptx")
                        presentation.run_slideshow()
                        presentation_opened = True
                        print("TRUE START")
                    else:
                        if other_gesture_after_opening_performed:
                            presentation.close_slideshow()
                            print("TRUE STOP")
                            break
                ### END START STOP GESTURE ###

            #End of "if confidences met".

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