import pyautogui
import pygetwindow as gw
import threading
from PIL import Image, ImageFilter, ImagePalette, ImageEnhance, ImageDraw
import numpy
from image2matrix import transform
from pathlib import Path
from astar_python.astar import Astar

counter = 0


def take_screenshot():
    global counter
    print('Taking screenshot', counter)
    windows = gw.getWindowsWithTitle('Path of Exile')
    if windows and windows[0].isActive:
        # windows[0].maximize()
        # windows[0].activate()

        leftPos = windows[0].topright.x - 260 - 6 - 8
        topPos = windows[0].topright.y + 32 + 4
        # print(
        #     leftPos,
        #     topPos,
        #     windows[0].box.top,
        #     windows[0].box.width,
        #     windows[0].width
        # )
        # time.sleep(1)
        im = pyautogui.screenshot(region=(leftPos, topPos, 260, 260))
        print(str(Path().absolute()))
        matrix = transform(im, save_debug_images=True, debug_image_name=str(counter), dir='./screenshots/')
        im.save('screenshots/{}.gif'.format(counter), 'gif')
        im.close()

        start = 130, 130
        # astar = Astar(matrix)
        # astar.run(start)

        # pyautogui.leftClick()

        # windows[0].minimize()
        counter += 1
    else:
        print('window is not active')


def set_timeout(func, interval_sec=3):
    func()
    threading.Timer(interval_sec, set_timeout, [func, interval_sec]).start()


if __name__ == '__main__':
    set_timeout(take_screenshot, 3)
