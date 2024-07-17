import numpy as np
import argparse
import cv2 as cv
import pytesseract
from pynput import mouse
import mss.tools
from process_vfr import *
from model import *
import json
from process_vfr import *
from collections import defaultdict

pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# Next steps:
# adaptability to different font sizes and spacing
# Use GCP to train on more than 20 fonts
# Improved usability

parser = argparse.ArgumentParser()
parser.add_argument("-d", action='store_true')
parser.add_argument("-l", action='store_true')
parser.add_argument("-t", action="store_true")
args = parser.parse_args()

dark_mode = args.d

# # from opencv docs, gaussian filtering can help w thresholding
# # Currently detects dark text on light background, but not light text on dark background
# # need some way to switch between dark/light mode
# # blur = cv.GaussianBlur(
# #     gray, (5,5), 0)
# # cv.imshow("blur", blur)
# # cv.waitKey(0)

# # Currently doing poorly on dynamic backgrounds, maybe need to look for gradients 
# # OR adaptive method
# activation = cv.THRESH_BINARY if dark_mode else cv.THRESH_BINARY_INV
# ret, thresh1 = cv.threshold(gray, 0, 255, cv.THRESH_OTSU | activation)
# # thresh1 = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 25, 5)

# # cv.imshow("thresh1", thresh1)
# # cv.waitKey(0)

# # Adjust kernel size to change size of retangle (detecting each sentence vs word)
# rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15)) # this needs to be different depending on font size

# # Getting rid of dilation means each character will be contoured
# # So far haven't been able to ignore specks with eroding
# # erode = cv.erode(thresh1, rect_kernel, iterations=1)
# dilation = cv.dilate(thresh1, rect_kernel, iterations=1)
# # dilation = thresh1
# # dilation = cv.morphologyEx(thresh1, cv.MORPH_OPEN, (25, 25))


# # cv.imshow("dilation", dilation)
# # cv.waitKey(0)

# # could try cv.CHAIN_APPROX_SIMPLE in place of chain_approx_none
# contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)


def pad_img(img):
    pad = [0,0,0] if dark_mode else [255,255,255]
    img = cv.copyMakeBorder(img, 15, 15, 15, 15, cv.BORDER_CONSTANT, value=pad)
    return img


def zoom_at(img, zoom=1, angle=0, coord=None):
    cy, cx = [ i/2 for i in img.shape[:] ] if coord is None else coord[::-1]
    
    rot_mat = cv.getRotationMatrix2D((cx,cy), angle, zoom)
    result = cv.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv.INTER_LINEAR)
    
    return result



    # Kaggle is really limiting my productivity, might be worth switching to gcp
    
    # use different config flags to search for words, single chars, etc
    # defaultConfig = "--psm 6"
    # singleConfig = "--psm 10" # Treat the image as a single character
    # autoConfig = "--psm 3" # "Fully automatic page segmentation, but no OSD"
    # sparseConfig = "--psm 12"
    
    # text = pytesseract.image_to_string(cropped, config=sparseConfig)

    # could use tesseract to read the text, and then display the text in each of the top fonts


def resize_image(img, image_dimension):
    """ Input: Image Path
    	Output: Image
    	Resizes image to height of 96px, while maintaining aspect ratio
    """
    base_height = image_dimension
    height_percent = (base_height/float(img.size[1]))
    wsize = int((float(img.size[0])*float(height_percent)))
    # print("Width", wsize)
    img = img.resize((wsize, base_height), Image.LANCZOS)

    return img


def get_screenshot():
    # sometimes the mouseclick down doesnt get recorded
    top = 0
    left = 0
    width = 0
    height = 0
    def on_click(x, y, button, pressed):
        nonlocal top, left, width, height
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
    while True:
        print("Capture photo")
        with mouse.Listener(on_click=on_click) as listener:
            listener.join() # have some feedback loop that shows the captured area
            with mss.mss() as sct:
                    monitor = {"top": top, "left": left, "width": width, "height": height}

                    # Grab the data
                    print("Captured screenshot")
                    sct_img = sct.grab(monitor)
                    img = np.array(sct_img)
                    cv.imshow("Look good?", img)
                    key = cv.waitKey(0)
                    # print(key)
                    cv.destroyAllWindows()
                    if key == ord('y'):
                        return img
                    return img

def get_top_5(img, model):
    altered = alter_image(img)
    resized = resize_image(altered, 96)
    squares = generate_crop(resized, 96, 5) 
    if len(squares) == 0:
        return []
    out = model.call(np.array(squares))
    total = np.sum(out, axis=0)

    # preds = np.argpartition(-total, 5)[:5]
    preds = np.argpartition(-total, 3)[:3]

    return preds


# having the tesseract option isnt a great solution, still need to make the choice
def tess_boxes(img):
    d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config="--psm 11")
    n_boxes = len(d['level'])
    boxes = []
    for i in range(n_boxes):
        # skip -1 confidence, those correspond with blocks of text
        if d["conf"][i] == -1:
            continue
        x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
        boxes.append((x, y, w, h))

    return boxes

def dilate_boxes(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dark_mode = args.d
    activation = cv.THRESH_BINARY if dark_mode else cv.THRESH_BINARY_INV
    ret, thresh1 = cv.threshold(gray, 0, 255, cv.THRESH_OTSU | activation)

    rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 11)) 

    dilation = cv.dilate(thresh1, rect_kernel, iterations=1)

    contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        boxes.append((x, y, w, h))
    return boxes

def use_model(model, get_bounds, filepath=None, repeat=1):
    if filepath:
        img = cv.imread(filepath)
    else:
        img = get_screenshot()
    contours = get_bounds(img)
    # lower contrast colors are a bit problematic
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    im2 = gray.copy()

    f = open('150_fonts_backwards.json')
    data = json.load(f)
    for x, y, w, h in contours:
        counts = defaultdict(int)
        # x, y, w, h = cv.boundingRect(cnt)

        cropped = im2[y:y + h, x:x + w] # can use the dimensions of cropped to figure out how much zoom is necessary
        cv.imshow("cropped", cropped)
        cv.waitKey(0)

        y_ratio = min(h / 96, 1)
        x_ratio = min(w / 96, 1)

        # might need to get crops here? could average from a couple crops
        pil = Image.fromarray(cropped)
        if x_ratio < 1 or y_ratio < 1:
            # pil = pad_img(cropped)
            pil = resize_image(pil, 96)

        for _ in range(repeat):
            preds = get_top_5(pil, model)
            if not len(preds):
                continue
            for pred in preds:  
                counts[data[str(pred)]] += 1
        print(counts)



def main():
    model = DeepFont()

    # call model to initialize weights
    model.call(np.ones((1,96,96,1)))
    # model.load_weights("weights/model_weights_20_leaky.weights.h5")
    model.load_weights("weights/model_weights_20_40epochs.weights.h5")

    print(np.argmax(model.call(np.ones((1,96,96,1))), axis=1))

    use_model(model, get_bounds=tess_boxes, repeat=10)
    filepath = "data/briem_space.png"
    # use_model(model, get_bounds=tess_boxes, filepath=filepath, repeat=100)

    # tesseract is adaptable, but it tends to draw too many boxes
    # img = cv.imread("data/briem_space.png")
    # img = cv.imread("data/antiqueOlive_close.png")
    # tess_boxes(img)
    

if __name__ == "__main__":
    main()


# Testing using top 3:
# Didn't perform well on baskerville cyrillic LT std
# Good on Americana std
# Ok on Arno pro
# Ok on Jenson pro
# Ok on Bauhaus std medium
# Ok on Bernhard Modern std roman
# Ok on Avenir lt std
# Good on Bookman std medium


# future idea: what if identifying color hexcodes lol

# model_weights_20_leaky - test accuracy 0.8
# model_weights_20_40epochs - training 0.92 accuracy, 0.33 loss, test accuracy 0.77
