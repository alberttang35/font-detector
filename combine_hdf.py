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




# ds = load_dataset("gaborcselle/font-examples", split="train", download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS)
# print(ds[:10]["image"])
# test = []
# for i in range(0,100):
#     test.append(get_samples(resize_image(ds[129]["image"], 105)).reshape((105, 105, 1)))
    
# print("loaded dataset")
# test = np.array(test)

# model = tf.keras.models.load_model('top_model.h5.keras')


# # get_samples(Image.fromarray(img))
# # print(img.size)
# # img = resize_image(img, 105)
# # img = get_samples(img).reshape((1, 105, 105, 1))
# # # img = np.array(img)
# # print(img.shape)
# out = model(test)
# print(out)

# softmax = np.apply_along_axis(np.argmax, arr=out, axis=1)
# print(softmax)

# f = open('font-codes-gaborcselle.json')
# data = json.load(f)

# def display(key):
#     print(data["names"][str(key)])


def combine_hdf(file1, file2):
    filepath1 = "data/" + file1 + ".hdf5"
    filepath2 = "data/" + file2 + ".hdf5"
    print("starting merge")

    with h5py.File(filepath1, 'r') as hf:
        ds1 = hf.get(file1)
        # arr = np.zeros((120000, 105, 105))
        arr1 = np.zeros((80000, ))
        ds1.read_direct(arr1)
        print(arr1[0])
        print(arr1[1])
        print(arr1[7999])
        print(arr1[8000])

    print("read file1")
    with h5py.File(filepath2, 'r') as hf:
        # print(hf)
        ds2 = hf.get(file2)
        arr2 = np.zeros((80000, 96, 96))
        ds2.read_direct(arr2)
        cv.imshow("test", arr2[0])
        cv.waitKey(0)
        cv.imshow("test", arr2[1])
        cv.waitKey(0)
        cv.imshow("test", arr2[7999])
        cv.waitKey(0)
        cv.imshow("test", arr2[8000])
        cv.waitKey(0)
        cv.imshow("test", arr2[8200])
        cv.waitKey(0)
        cv.imshow("test", arr2[16000])
        cv.waitKey(0)
    print("read file2")
    # aggregate = np.empty((arr1.shape[0] + arr2.shape[0], 105, 105))
    # aggregate = np.concatenate((arr1, arr2))
    # print("allocated aggregate")
    # print(aggregate.shape)
    # aggregate[:arr1.shape[0],:,:] = arr1.copy()
    # aggregate[arr1.shape[0]:,:,:] = arr2.copy()

    # print(aggregate[68129])
    # print(arr1[68129])

    # print("------------")

    # print(aggregate[160000])
    # print(arr2[0])
    # name = "train_inputs_12"
    # with h5py.File(name + "hdf5", 'w') as f:
    #     f.create_dataset(name,data=aggregate,compression="gzip", compression_opts=9)

def test_model():
    batch_size = 128
    model = DeepFont(batch_size=batch_size)
    model.load_weights("model_weights.weights.h5")
    model.summary()
    # with h5py.File("data/toy_train_inputs.hdf5", 'r') as hf:
    #     ds = hf.get("toy_train_inputs")
    #     arr2 = np.zeros((80000, 96, 96))
    #     ds.read_direct(arr2)
    # pred = model(arr2)
    # print(pred)


def main():
    combine_hdf("toy_train_labels", "toy_train_inputs")
    # test_model()

if __name__ == "__main__":
    main()