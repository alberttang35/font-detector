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
from heapq import nlargest

pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# Next steps:
# Use GCP to train on more than 20 fonts
# Improved usability

parser = argparse.ArgumentParser()
parser.add_argument("-d", action='store_true', help="Dark mode. Use for light text on dark background")
parser.add_argument("-v", "--verbose", action='store_true')
parser.add_argument("--filepath", type=str, default=None, help="Filepath to detect fonts. If none provided you will be prompted to take a picture")
parser.add_argument("--top", type=int, default=3, help="Number of predictions to consider for each crop")
parser.add_argument("--model", choices=['TFLite', 'Compact', 'Full', 'Checkpoint-50', 'Checkpoint-149'], default='Compact')
parser.add_argument("--border", choices=['b', 'w', None], default=None)
args = parser.parse_args()

# verbose = False
split = args.top
dark_mode = args.d

# # from opencv docs, gaussian filtering can help w thresholding
# # Currently detects dark text on light background, but not light text on dark background
# # need some way to switch between dark/light mode
# # blur = cv.GaussianBlur(
# #     gray, (5,5), 0)

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


def zoom_at(img, zoom=1, angle=0, coord=None):
    cy, cx = [ i/2 for i in img.shape[:] ] if coord is None else coord[::-1]
    
    rot_mat = cv.getRotationMatrix2D((cx,cy), angle, zoom)
    result = cv.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv.INTER_LINEAR)
    
    return result

    
    # text = pytesseract.image_to_string(cropped, config=sparseConfig)
    # could use tesseract to read the text, and then display the text in each of the top fonts


def resize_image(img, image_dimension):
    """ Input: Image, image_dimension
    	Output: Image
    	Resizes image to height of image_dimension, while maintaining aspect ratio
    """
    base_height = image_dimension
    height_percent = (base_height/float(img.size[1]))
    wsize = int((float(img.size[0])*float(height_percent)))
    img = img.resize((wsize, base_height), Image.LANCZOS)

    return img


def get_screenshot() -> np.ndarray:
    '''
    Take a screenshot to infer fonts
    '''
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
                    print("Press y to confirm this capture. Press any other button to retake")
                    key = cv.waitKey(0)
                    cv.destroyAllWindows()
                    cv.waitKey(1)
                    if key == ord('y'):
                        return img
                    # return img
            
def get_preds(squares, model):
    '''
    Get the model's predictions for each crop
    Args:
        squares - 
        model - 
    '''
    if model:
        out = model.call(np.array(squares))
        total = np.sum(out, axis=0)
        return total
    else:
        # use the tflite model
        interpreter = tf.lite.Interpreter(model_path="weights/model.tflite")
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_shape = input_details[0]['shape']
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        out = []
        for crop in squares:
            interpreter.set_tensor(input_details[0]['index'], np.array(crop, dtype=np.float32).reshape(1, 96, 96))

            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            out.append(output_data[0])

        total=np.sum(np.array(out), axis=0)
        return total


def get_top_5(img, model):
    '''
    Processes img and gets the model's top five predictions for its font
    '''
    altered = alter_image(img)
    resized = resize_image(altered, 96)
    squares = generate_crop(resized, 96, 5) 
    if len(squares) == 0:
        return []
    
    total = get_preds(squares, model)
    
    preds = np.argpartition(-total, split)[:split]

    return preds

def tess_boxes(img):
    '''
    Use tesseractOCR to detect bounding boxes for each word
    '''
    activation = cv.THRESH_BINARY if dark_mode else cv.THRESH_BINARY_INV
    # Not sure if thresholding helps
    ret, thresh1 = cv.threshold(img, 0, 255, cv.THRESH_OTSU | activation)
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
    '''
    Use openCV to find bounding boxes for each word
    '''
    dark_mode = args.d
    activation = cv.THRESH_BINARY if dark_mode else cv.THRESH_BINARY_INV

    ret, thresh1 = cv.threshold(img, 0, 255, cv.THRESH_OTSU | activation)
    # cv.imshow("thresh", thresh1)
    # cv.waitKey(0)
    kernel_shape = (15, 11)
    rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_shape) 

    dilation = cv.dilate(thresh1, rect_kernel, iterations=1)

    contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        boxes.append((x, y, w, h))
    return boxes

def use_model(model, get_bounds, topN:int, filepath=None, repeat=1):
    '''
    Run the font inferrence model
    '''
    if filepath:
        img = cv.imread(filepath)
    else:
        img = get_screenshot()
    

    # doesnt work too well on mixed background
    
    # One idea for dealing with concentric bg
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if args.border:
        fill = 0 if args.border == 'w' else 255
        _, gray, _, _ = cv.floodFill(gray, None, (0,0), fill)

    contours = get_bounds(gray)

    # Thresholding helps with non-black/white font colors
    activation = cv.THRESH_BINARY_INV if dark_mode else cv.THRESH_BINARY # This gets black text on white bg
    ret, thresh1 = cv.threshold(gray, 0, 255, cv.THRESH_OTSU | activation) # does the model care if its white text or black text?

    im2 = thresh1.copy()

    f = open('utils/149_fonts_backwards.json')
    data = json.load(f)
    counts = defaultdict(int)
    for x, y, w, h in contours:

        cropped = im2[y:y + h, x:x + w] 

        if args.verbose:
            cv.imshow("crop", cropped)
            cv.waitKey(100)

        y_ratio = h / 96
        x_ratio = w / 96

        pil = Image.fromarray(cropped)
        if x_ratio < 1 or y_ratio < 1:
            pil = resize_image(pil, 96)

        for _ in range(repeat):
            preds = get_top_5(pil, model)
            if not len(preds):
                continue
            for pred in preds:  
                counts[data[str(pred)]] += 1
    # get top n
    return nlargest(topN, counts, key=counts.get)

def load_checkpoint(model, ckpt):
    checkpoint = tf.train.Checkpoint(model = model)
    checkpoint.restore(ckpt).expect_partial()

def main():
    if args.model == 'Full':
        model = DeepFont(512)
        model.call(np.ones((1,96,96,1)))
        model.load_weights("weights/model_weights_20_40epochs.weights.h5")
        model.call(np.ones((1,96,96,1)))
    elif args.model == 'Compact':
        model = DeepFont(128)
        model.call(np.ones((1,96,96,1)))
        model.load_weights("weights/model_weights_20_compact.weights.h5")
        model.call(np.ones((1,96,96,1)))
    elif args.model == 'Checkpoint-50':
        # 75% test accuracy
        model = DeepFont(512, 50)
        model.call(np.ones((1,96,96,1)))
        load_checkpoint(model, "checkpoints_df/ckpt-55")
        model.call(np.ones((1,96,96,1)))
    elif args.model == 'Checkpoint-149':
        # Currently about 78% test accuracy
        model = DeepFont(512, 149)
        model.call(np.ones((1,96,96,1)))
        load_checkpoint(model, "checkpoints_df/ckpt-56")
        model.call(np.ones((1,96,96,1)))
    else:
        model = None # this will activate the TFLite model
    

    filepath = args.filepath
    
    res = use_model(model, get_bounds=tess_boxes, topN=3, filepath=filepath, repeat=100)

    print(res)
    

if __name__ == "__main__":
    main()



# future idea: what if identifying color hexcodes lol

# model_weights_20_leaky - test accuracy 0.8
# model_weights_20_40epochs - training 0.92 accuracy, 0.33 loss, test accuracy 0.77
