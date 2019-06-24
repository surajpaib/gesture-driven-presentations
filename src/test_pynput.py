from pynput.keyboard import Key, Controller
keyboard = Controller()
keyboard.press(Key.f12)
keyboard.release(Key.f12)