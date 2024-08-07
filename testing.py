import multiprocessing.queues
from typing import List
import numpy as np
from PIL import Image
import cv2 as cv
import json
import multiprocessing
import time
import websocket
import base64
import io
import argparse

# TODO: make experience with camera more smooth, maybe see the frame sent to gpt
# TODO: is this good separation of concerns?

parser = argparse.ArgumentParser()
parser.add_argument('-s', action='store_true', help='If present, images are streamed to ChatGPT for inference')
parser.add_argument('--query', '-q', help='USAGE: --query="<query sentence>"') # how to get query from user
args = parser.parse_args()


def generate_test():
    '''
    Basic example for accessing the stream of inferences from infer_stream
    '''
    for value in infer_stream():
        print(value)

def infer_stream():
    '''
    Opens camera and streams frames to GPT4 for inference
    '''
    q = multiprocessing.Queue()
    proc = multiprocessing.Process(name="use_camera", target=use_camera, args=[q])
    proc.start()
    try:
        while proc.is_alive(): 
            if not q.empty():
                data = q.get()

                if type(data) is str and data == "STOP":
                    break
                inference = get_inference(data, 3243232, "")
                yield inference
            else:
                time.sleep(1)
    except Exception as e:
        # Allowed to join child proc because child calls cancel_join_thread
        proc.join()
        proc.close()
        raise e

    proc.join()
    proc.close()
    

def simple_infer(chat_id: int, query = "") -> str:
    '''
    Get a single image from local camera and a summary from GPT4
    Args:
        chat_id - 
        query - 
    '''
    print("Press q to capture image for inference") # indicate how to use
    # frame = use_camera()
    frame = None
    out = get_inference(frame, chat_id, query)
    return out
    

def use_camera(q = None) -> np.ndarray:
    '''
    Opens local camera for capture. Returns an ndarray image
    Args:
        q - Optional queue for sending frames to another process
    '''
    if type(q) is not multiprocessing.queues.Queue and q is not None:
        raise Exception("q parameter has invalid type")
    vid = cv.VideoCapture(0)
    if not vid.isOpened():
        raise Exception("Cannot open camera")
    count = 0

    try:
        ret, frame = vid.read()
        while not ret:
            ret, frame = vid.read() # could this get stuck?

        while ret: 
            count += 1

            if q and count % 200 == 0: # this might be more interesting if it triggered based on an event happening
                q.put(frame)

            cv.imshow('frame', frame)  
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            ret, frame = vid.read()
        if q:
            # q.put(frame)
            q.put("STOP")
    except Exception as e:
        q.cancel_join_thread()
        raise e
    vid.release()

    if q:
        q.cancel_join_thread()

    if frame is None:
        raise Exception("Unexpected disconnect from camera")

    cv.destroyAllWindows()
    cv.waitKey(1)
    return frame

def get_inference(frame: np.ndarray, chat_id: int, query: str) -> str:
    '''
    Connects to websocket and gets an inference for frame
    Args:
        frame - Image to get inference from
        chat_id - 
        query - Optional question or comment to accompany the image for inference
    '''
    # I like the organization now because it is easy to understand
    # However, each part isn't really usable by itself. For example, you wouldn't use send_image by itself
    # Which makes me want to refactor
    try:
        url = "ws://172.21.1.104:18075/v1/public/generat"
        ws = websocket.create_connection(url)
        response = send_image(ws, frame, chat_id, query)
        return response
    except Exception as e:
        return "failed to connect to inference server"

def send_image(ws, frame: np.ndarray, chat_id: int, query: str) -> str:
    '''
    Sends frame to websocket ws for inference. Returns the result of inference
    Args:
        ws - Websocket connection
        frame - Image to send
        chat_id - 
        query - 
    '''
    img = Image.fromarray(frame)

    img_str = encode_image(img)
    img_str = 'data:image/jpeg;base64,'+img_str

    message = {
        "chat_id": chat_id, # dont want to always use the same id
        "query_text": query, # might want to ask a specific question
        "model_name": "gpt4",
        "image_info": [img_str] 
    }
    message = json.dumps(message)

    ws.send(message)
    output = ""
    data = ws.recv()
    while data:
        output += data
        data = ws.recv()
    # slackbot handles the issue of recv -> disconnect by reconnecting after reading
    return output

def encode_image(img: Image) -> str:
    '''
    Returns img as a base64 encoded, utf8 string
    '''
    # Make sure image is in RGB
    img = img.convert('RGB')
    # Limit dimensions to (512, 512)
    img.thumbnail((512, 512))
    byte_arr = io.BytesIO()
    img.save(byte_arr, format="PNG")
    img_str = base64.b64encode(byte_arr.getvalue()).decode("utf-8")
    return img_str


def main():
    print("main")
    if args.s:
        # infer_stream()
        generate_test()
    else:
        print(simple_infer(chat_id=3243232, query=args.query))
    


if __name__ == "__main__":
    main()