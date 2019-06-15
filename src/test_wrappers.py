import time
import win32com.client
import pyautogui

from wrappers import PowerpointWrapper, PresentationWrapper

# First, create a PowerpointWrapper. This should wrap most of the functions
# from the ugly Powerpoint API.
wrapper = PowerpointWrapper()

# Then, you can use the wrapper to create a PresentationWrapper.
presentation = wrapper.open_presentation("../MRP-6.pptx")

# Next calls are really self-explanatory.
presentation.run_slideshow()
time.sleep(5)
presentation.next_slide()
presentation.start_spotlight()
time.sleep(5)
presentation.start_zoom()
time.sleep(3)
presentation.stop_zoom()
time.sleep(3)
presentation.close_slideshow()

# Quit the Powerpoint application.
wrapper.quit()
