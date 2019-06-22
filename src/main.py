import time
from typing import Tuple

import imutils

from ml.autoencoder import *
from ml.classifier import Classifier
from ml.correlation_classifier import CorrelationClassifier
from openpose_utils import *
from video_processing.frame_data import FrameData
from video_processing.keypoints import get_all_keypoints_list
from video_processing.video_data import VideoData
from wrappers import PowerpointWrapper

current_milli_time = lambda: int(round(time.time() * 1000))


def extract_hand_regions(frame: np.ndarray, hand_rectangles: List[op.Rectangle]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crops out the two hand regions from the image, based on the given rectangles.
    """

    # Extract left hand rectangle if it was detected.
    if hand_rectangles[0].x != 0:
        rect_x_1 = int(hand_rectangles[0].x)
        rect_y_1 = int(hand_rectangles[0].y)
        rect_x_2 = int(hand_rectangles[0].x + hand_rectangles[0].width)
        rect_y_2 = int(hand_rectangles[0].y + hand_rectangles[0].height)
        left_hand_region = frame.copy()[rect_y_1:rect_y_2, rect_x_1:rect_x_2]
    else:
        left_hand_region = None

    # Extract right hand rectangle if it was detected.
    if hand_rectangles is not None and hand_rectangles[1].x != 0:
        rect_x_1 = int(hand_rectangles[1].x)
        rect_y_1 = int(hand_rectangles[1].y)
        rect_x_2 = int(hand_rectangles[1].x + hand_rectangles[1].width)
        rect_y_2 = int(hand_rectangles[1].y + hand_rectangles[1].height)
        right_hand_region = frame.copy()[rect_y_1:rect_y_2, rect_x_1:rect_x_2]
    else:
        right_hand_region = None

    # Display the two rectangles.
    if left_hand_region is not None and left_hand_region.shape[0] > 0 and left_hand_region.shape[1] > 0:
        cv2.imshow("Left hand region", left_hand_region)
    if right_hand_region is not None and right_hand_region.shape[0] > 0 and right_hand_region.shape[1] > 0:
        cv2.imshow("Right hand region", right_hand_region)

    # Return the two regions.
    return left_hand_region, right_hand_region


def hand_segmentation(left_hand_region: np.ndarray, right_hand_region: np.ndarray):
    # TODO
    pass


# -------------------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize OpenPose.
    opWrapper = init_openpose(hand_detection=False)

    # Get the reference to the webcam.
    camera = cv2.VideoCapture(0)
    num_frames = 0

    presentation_opened = False
    wrapper = None

    # Used for basic wrist gesture detection.
    r_wrist_x_old = 0
    l_wrist_x_old = 0

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

        if keypoints is not None:
            # Extract the hand rectangles, segment out the hand and perform some gesture detection (?).
            if hand_rectangles:
                left_hand_region, right_hand_region = extract_hand_regions(frame, hand_rectangles)
                hand_segmentation(left_hand_region, right_hand_region)

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

        # If the user pressed "Q", then quit.
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break

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
