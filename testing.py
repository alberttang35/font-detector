import numpy as np
from PIL import Image
import cv2 as cv
import random
from process_vfr import *
import json
import requests
import h5py
from model import *
import pytesseract
from multiprocessing import Process, Pool, Queue
import time
import websocket
import base64
import io
import hashlib
from threading import Thread


def use_model(model, q):
    # Could reverse design so that this function is main and camera is subprocess
    # This way it is easier to access the results of model inference
    boxes = []
    print("child created")
    # ws = websocket.WebSocket()
    while True:
        if not q.empty():
            data = q.get()
            if type(data) is str and data == "STOP":
                return
            # d = pytesseract.image_to_data(data, output_type=pytesseract.Output.DICT, config="--psm 11")
            # n_boxes = len(d['level'])
            # for i in range(n_boxes):
            #     if d['conf'][i] == -1:
            #         continue
            #     print(d['text'][i])
            
            d = pytesseract.image_to_string(data, config="--psm 11")
            print(d)
            # for any model, this data d must go somewhere, but how?
        else:
            time.sleep(1)
    
    

# problem: call to the model is blocking, causing video to buffer

def use_camera():
    vid = cv.VideoCapture(0)
    if not vid.isOpened():
        print("Cannot open camera")
        return
    count = 0
    boxes = []

    # use this process to handle calling the model so that the main proc doesnt block
    model = None
    q = Queue()
    proc = Process(target=use_model, args=(model, q))
    proc.start()
    

    ret, frame = vid.read()
    while ret:
        boxes = []
        
        count += 1
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)
        if count % 50 == 0:
            q.put(thresh)
    
        # for x, y, w, h in boxes:
        #     cv.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
        cv.imshow('frame', thresh)  
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        # count += 1
        ret, frame = vid.read()
  

    vid.release()
    q.put("STOP")
    q.cancel_join_thread()
    proc.join()
    proc.close()
    cv.destroyAllWindows()


def connect_ws():
    # img = cv.imread("data/test-data.png")

    # pil = Image.fromarray(img)
    # pil_bytes = base64.b64encode(pil.tobytes())
    # byte_arr = io.BytesIO()
    # pil.save(byte_arr, format="PNG")

    # pil_bytes = base64.b64encode(byte_arr.getvalue())
    # # print(byte_arr.getvalue())
    # # byte_arr = io.BytesIO(pil)
    # b64_encoded_ascii = pil_bytes.decode("ascii")
    # b64_encoded = pil_bytes.decode("utf-8")

    with Image.open("data/test-data.png") as img:
        img = img.convert('RGB')
        img.thumbnail((512, 512))
        byte_arr = io.BytesIO()
        img.save(byte_arr, format="PNG")
        img_str = base64.b64encode(byte_arr.getvalue()).decode("utf-8")
        img_str = 'data:image/jpeg;base64,'+img_str


    # with open("data/antiqueOliveStd.png", 'rb') as imgfile:
    #     b64_bytes = base64.b64encode(imgfile.read())
    #     b64_encoded = b64_bytes.decode('utf-8')
    #     b64_encoded = 'data:image/jpeg;base64,' + b64_encoded

    # Something about message is causing the websocket to error and close immediately
    message = {
        "chat_id": "3243232",
        # "query_text": "I submitted an image for inferrence. What is it?",
        # "query_text": "If there is an error, please return it",
        "query_text": "",
        # "query_text": None,
        "model_name": "gpt4",
        "image_info": [img_str] 
    }
    message = json.dumps(message)
    
    def on_message(ws, message):
        try:
            # content = json.loads(message)
            # print("CONTENT")
            # print(content)
            print("RECEIVED: " + message)

            # byt = base64.b64decode(content["image_info"][0])
            # print(byt)
            # byte_arr = io.BytesIO(bytearray(byt))
            # img = Image.open(byte_arr)
            # print("processed image")
            # cv.imshow("test", np.asarray(img))
            # cv.waitKey(1)
            
        except Exception as e:
            print(e)
            print("RECEIVED: " + message)


    def on_error(ws, error):
        print("ERROR: " + error)

    def on_close(ws, close_status_code, close_msg):
        print("CLOSED")
        print(close_status_code)

    def on_open(ws):
        print("sending message")
        ws.send(message)


    url = "ws://172.21.1.104:18075/v1/public/generate"
    ws = websocket.WebSocketApp(url, on_close=on_close, on_error=on_error, 
                                on_open=on_open, on_message=on_message)
    ws.run_forever()

    # jsondict = json.loads(message)
    # to_decode = jsondict["image_info"]

    # to_decode.encode()
    # for elt in to_decode:
    #     decode_ascii = elt.encode("utf-8")
    #     # print(decode_ascii)
    #     # pil_byte = base64.b64decode(decode_ascii) # The invalid base64 error comes from here
    #     pil_byte = base64.decodebytes(decode_ascii)

    #     byte_arr = io.BytesIO(bytearray(pil_byte))
    #     # base64.decode(to_decode, byte_arr)
    #     img = Image.open(byte_arr)
    #     cv.imshow("test", np.asarray(img))
    #     cv.waitKey(0)

    ws.close()


def main():
    print("main")
    # use_camera()
    connect_ws()


if __name__ == "__main__":
    main()