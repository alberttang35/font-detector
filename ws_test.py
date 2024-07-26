import numpy as np
from PIL import Image
import cv2 as cv
from datasets import load_dataset, DownloadMode
import random
# from keras.models import load_model
import tensorflow as tf
from process_vfr import *
import json
import requests
import h5py
from model import *
import pytesseract
from multiprocessing import Process, Pool, Queue
import threading
import time
import websocket
import base64
import io
import hashlib

def on_open(ws):
    ws.send("Hello")

def on_close(ws, close_status_code, close_msg):
    print("close")

def on_error(ws, error):
    print(error)

def on_data(ws, message):
    print(message)

def main():
    ws = websocket.WebSocketApp("ws://localhost:8888", on_open=on_open, on_close=on_close, on_data=on_data, on_error=on_error)
    ws.run_forever()

if __name__ == "__main__":
    main()