import os
import win32com.client
import pyautogui

class PresentationWrapper():
    def __init__(self, presentation):
        self.presentation = presentation
        self.spotlight = False
        self.zoom = False

    def run_slideshow(self):
        self.presentation.SlideShowSettings.Run()

    def close_slideshow(self):
        self.presentation.SlideShowWindow.View.Exit()

    def next_slide(self):
        self.presentation.SlideShowWindow.View.Next()
    
    def previous_slide(self):
        self.presentation.SlideShowWindow.View.Previous()

    def start_spotlight(self):
        """
        Hacky right now. It presses F10 which starts the spotlight function
        from PointerFocus.
        """
        # Stop zoom if it is started.
        if self.zoom:
            pyautogui.press('f12')
            self.zoom = False

        # Do nothing if spotlight already started.
        if self.spotlight:
            return

        self.spotlight = True
        pyautogui.press('f10')

    def stop_spotlight(self):
        """
        Hacky right now. It presses F10 which stops the spotlight function
        from PointerFocus.
        """
        # Do nothing if spotlight is not running.
        if not self.spotlight:
            return

        self.spotlight = False
        pyautogui.press('f10')

    def start_zoom(self):
        """
        Hacky right now. It presses F12 which starts the zoom function
        from PointerFocus.
        """
        # Stop spotlight if it is started.
        if self.spotlight:
            pyautogui.press('f10')
            self.spotlight = False

        # Do nothing if zoom is already running.
        if self.zoom:
            return

        self.zoom = True
        pyautogui.press('f12')

    def stop_zoom(self):
        """
        Hacky right now. It presses F12 which stops the zoom function
        from PointerFocus.
        """
        # Do nothing if zoom is not running.
        if not self.zoom:
            return

        self.zoom = False
        pyautogui.press('f12')

class PowerpointWrapper():
    def __init__(self):
        self.application = None

    def start_powerpoint(self):
        if self.application is None:
            self.application = win32com.client.Dispatch("PowerPoint.Application")
            self.application.Visible = 1
    
    def open_presentation(self, filename) -> PresentationWrapper:
        if self.application is None:
            self.start_powerpoint()

        presentation = self.application.Presentations.Open(os.path.abspath(filename))
        return PresentationWrapper(presentation)

    def quit(self):
        self.application.Quit()
