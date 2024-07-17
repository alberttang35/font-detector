import numpy as np
from PIL import ImageFilter
import argparse
import keras
import os
from keras import optimizers
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras import callbacks
from keras.models import Sequential, load_model
from keras import backend as K
import tensorflow as tf
from process_vfr import *
from datasets import load_dataset
# from keras.preprocessing.image import ImageDataGenerator

parser = argparse.ArgumentParser()
parser.add_argument("-r", action='store_true')
args = parser.parse_args()


K.set_image_data_format('channels_last')
def create_model():
  model=Sequential()

#   model.add(keras.Input(shape=(105, 105, 1))) # not sure if this is whats causing hanging

  # Cu Layers 
  model.add(keras.layers.Conv2D(64, kernel_size=(48, 48), activation='relu', input_shape=(105, 105, 1)))
  model.add(keras.layers.BatchNormalization())
  model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

  model.add(keras.layers.Conv2D(128, kernel_size=(24, 24), activation='relu'))
  model.add(keras.layers.BatchNormalization())
  model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))


    # i dont think these layers are from the paper
  model.add(keras.layers.Conv2DTranspose(128, (24,24), strides = (2,2), activation = 'relu', padding='same', kernel_initializer='uniform'))
  model.add(keras.layers.UpSampling2D(size=(2, 2)))

  model.add(keras.layers.Conv2DTranspose(64, (12,12), strides = (2,2), activation = 'relu', padding='same', kernel_initializer='uniform'))
  model.add(keras.layers.UpSampling2D(size=(2, 2)))

  #Cs Layers
  model.add(keras.layers.Conv2D(256, kernel_size=(12, 12), activation='relu'))

  model.add(keras.layers.Conv2D(256, kernel_size=(12, 12), activation='relu'))

  model.add(keras.layers.Conv2D(256, kernel_size=(12, 12), activation='relu'))

  model.add(keras.layers.Flatten())

  model.add(keras.layers.Dense(4096, activation='relu'))

  model.add(keras.layers.Dropout(0.5))

  model.add(keras.layers.Dense(4096,activation='relu'))

  model.add(keras.layers.Dropout(0.5))

  model.add(keras.layers.Dense(2383,activation='relu'))

  model.add(keras.layers.Dense(4, activation='softmax')) # number of possible classifications
 
  return model

batch_size = 128
epochs = 20
model= create_model()
if args.r:
  model = tf.keras.models.load_model("top_model.h5.keras")

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
# adam = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min')

# aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,horizontal_flip=True)


filepath="top_model.h5.keras"

checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [early_stopping,checkpoint]

ds = load_dataset("gaborcselle/font-examples", split="train")

imgs = []
labels = []
# can this be vectorized? its gotta be pretty slow
# print(type(ds)) # might be able to directly take the slice I want instead of this loop
for i in range(200):
  # rows = ds[:250,:250]
  # labels.extend([rows["label"]] * 5)
  # imgs.extend(get_samples(rows["image"]))
  row = ds[i]
  labels.extend([row["label"]] * 4)
  imgs.extend(get_samples(row["image"], 4))
  # count = imgs.extend(get_samples(row["image"]))
  # labels.extend([row["label"]] * count)

dataX = np.array(imgs) / 255.0
# print(dataX)
dataY = np.array(labels)


# this is shuffling the data, not sure why my val accuracy and loss is so bad
trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size=0.1, random_state=1)

trainY = to_categorical(trainY, num_classes=4)
testY = to_categorical(testY, num_classes=4)

model.fit(trainX, trainY,shuffle=True,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(testX, testY),callbacks=callbacks_list)
score = model.evaluate(testX, testY, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# print("testing")
# validation = ds[1199]["image"]
# print(validation.size)
# sampled = get_samples(validation).reshape((1, 105, 105, 1))
# print(sampled.shape)
# print(np.array(sampled).shape)
# # score = model.predict(np.array(sampled))
# score = model(np.array(sampled))
# print("Score: ", score)

# what isnt working
# loss and val_loss being so low feels wrong considering low accuracy

# if this run doesnt work, then i'll take a closer look at the preprocessing