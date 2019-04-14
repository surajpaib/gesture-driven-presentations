# organize imports
import cv2
import imutils
import numpy as np

from wrappers import PowerpointWrapper, PresentationWrapper
from sklearn.metrics import pairwise

from openpose_keypoints import *

# global variables
bg = None

#-------------------------------------------------------------------------------
# Main function
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize openpose
    opWrapper = init_openpose()

    # initialize accumulated weight
    accumWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 25, 350, 225, 550

    # initialize num of frames
    num_frames = 0

    # calibration indicator
    calibrated = False

    # video recorder
    video_writer = cv2.VideoWriter('testvideo.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, 
           (640,480),True)

    # keep looping, until interrupted

    presentation_opened = False

    r_wrist_x_old = 0
    l_wrist_x_old = 0
    print("START")
    counter = 0
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()
        video_writer.write(frame)

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame_flipped = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        r_wrist, l_wrist = getKeypointsFromImage(frame, opWrapper)
        r_wrist_x = r_wrist[0]
        r_wrist_y = r_wrist[1]
        l_wrist_x = l_wrist[0]
        l_wrist_y = l_wrist[1]
        #print("RWrist: "+str(r_wrist_x))
        #print("LWrist: "+str(l_wrist_x))
        action = ""
        if (r_wrist[2] > 0.3) and (l_wrist[2] > 0.3):
            if num_frames < 30:
                r_wrist_x_old = r_wrist_x
                l_wrist_x_old = l_wrist_x
                #if num_frames == 1:
                #    print("[STATUS] please wait! calibrating...")
                #elif num_frames == 29:
		    	#    print("[STATUS] calibration successfull...")   
                pass    
            else:
                # Open presentation and stuff:
                if(l_wrist_x > -1):
                    print("L: "+str(l_wrist_x_old) + "  "+ str(l_wrist_x))
                    cv2.circle(clone, (l_wrist_x, l_wrist_y), 10, (0,0,255), -1)
                    if(l_wrist_x_old - l_wrist_x > 150):
                        if not presentation_opened:
                            action = "OPEN"
                            wrapper = PowerpointWrapper()
                            presentation = wrapper.open_presentation("../MRP-6.pptx")
                            presentation.run_slideshow()
                            presentation_opened = True
                        else:
                            action = "PREV"
                            presentation.previous_slide()
                    l_wrist_x_old = l_wrist_x
                if(r_wrist_x > -1):
                    print("R: "+str(r_wrist_x_old) + "  "+ str(r_wrist_x))
                    cv2.circle(clone, (r_wrist_x, r_wrist_y), 10, (255,0,0), -1)
                    if(r_wrist_x - r_wrist_x_old > 150):
                        if not presentation_opened:
                            action = "OPEN"
                            wrapper = PowerpointWrapper()
                            presentation = wrapper.open_presentation("../MRP-6.pptx")
                            presentation.run_slideshow()
                            presentation_opened = True
                        else:
                            presentation.next_slide()
                            action="NEXT"
                    r_wrist_x_old = r_wrist_x
            cv2.putText(clone, str(action), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break
        elif keypress == ord("r"):
            num_frames = 0

        counter += 1

# free up memory
camera.release()
video_writer.release()
cv2.destroyAllWindows()
