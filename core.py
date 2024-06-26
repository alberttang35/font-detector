import numpy as np
import argparse
import cv2 as cv
import pytesseract
from pynput import mouse
import mss.tools

# How do you go about classifying a font?
# Need to compare like characters. Eg compare "A" to "A", not "A" to "B"
# This requires identifying characters, which shouldn't be too hard since we're dealing with typed characters
# I think pytesseract can be used to identify characters

# Whatever dataset used, need to be able to recreate the shape of that data for use outside of training/testing

pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

top = 0
left = 0
width = 0
height = 0
def on_click(x, y, button, pressed):
    global top, left, width, height
    if pressed:
        top = y
        left = x
    if not pressed: 
        width = abs(x - left)
        left = min(x, left)
        height = abs(y - top)
        top = min(y, top)
        if width > 10 and height > 10: # otherwise, dont consider the event a "drag"
            return False    

with mouse.Listener(on_click=on_click) as listener:
    listener.join()
    with mss.mss() as sct:
        monitor = {"top": top, "left": left, "width": width, "height": height}
        output = "sct-{top}x{left}_{width}x{height}.png".format(**monitor)

        # Grab the data
        sct_img = sct.grab(monitor)

        # Save to the picture file
        # mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
        # cv.imshow("OpenCV/test", np.array(sct_img))
# print(sct_img)

img = np.array(sct_img)

# img = cv.imread("data/test-py.png")

# TODO: better pre-processing for light text/dark mode

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow("gray", gray)
cv.waitKey(0)

ret, thresh1 = cv.threshold(gray, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)

# Adjust kernel size to change size of retangle (detecting each sentence vs word)
# kernel size should change with size of image
rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (8, 8))

dilation = cv.dilate(thresh1, rect_kernel, iterations=1)

contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

im2 = img.copy()

file = open("recognized.txt", "w+")
file.write("")
file.close()

for cnt in contours:
    print(cnt)
    print(1)
    x, y, w, h = cv.boundingRect(cnt)

    rect = cv.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow("test", rect)
    cv.waitKey(0)

    cropped = im2[y:y + h, x:x + w]

    file = open("recognized.txt", "a")

    
    # use different config flags to search for words, single chars, etc
    defaultConfig = "--psm 6"
    singleConfig = "--psm 10"
    text = pytesseract.image_to_string(cropped, config=defaultConfig)
    print(cropped.shape)
    print(text)

    # Use text to figure out which characters to compare

    file.write(text)
    file.write("\n")
    file.close()

# Hello world!


# A




#   Im a robot