import numpy as np
from PIL import Image
import cv2 as cv
# from imutils import paths
# import itertools
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose
from datasets import load_dataset
import os
from process_vfr import *
from sklearn.model_selection import train_test_split
import argparse
import keras



# parser = argparse.ArgumentParser()
# parser.add_argument("--restore-checkpoint", dest="restore", action='store_true')
# parser.add_argument("--mode", dest="mode", type=str, default="train")
# parser.add_argument("--batch-size", dest="batch_size", type=int, default=128)
# parser.add_argument("--num-epochs", dest="num_epochs", type=int, default=10)

# args = parser.parse_args()

#--------------------------------------------------------------------------------


performance_dict = {}

class DeepFont(Model):
    def __init__(self, dense=128, num_classes=20):
        super(DeepFont, self).__init__()

        self.batch_size = 128
        self.num_classes = num_classes
        self.leaky_relu = tf.keras.layers.LeakyReLU(negative_slope=0.2)

        self.model = tf.keras.Sequential()
        # C_u from Deepfont
        self.model.add(tf.keras.layers.Reshape((96, 96, 1)))
        self.model.add(tf.keras.layers.Conv2D(trainable=False, filters=64, strides=(2,2), kernel_size=(3,3), padding='same', name='conv_layer1'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=None, padding='same'))
        
        self.model.add(tf.keras.layers.Conv2D(trainable=False, filters=128, strides=(1,1), kernel_size=(3,3), padding='same', name='conv_layer2'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=None, padding='same'))

        # C_s from Deepfont
        self.model.add(tf.keras.layers.Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same'))
        self.model.add(tf.keras.layers.Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same'))
        self.model.add(tf.keras.layers.Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same'))

        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(dense, activation=self.leaky_relu))
        self.model.add(tf.keras.layers.Dropout(0.4))
        self.model.add(tf.keras.layers.Dense(dense, activation=self.leaky_relu))
        self.model.add(tf.keras.layers.Dropout(0.4))
        self.model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax')) 


    def call(self, inputs):

        return self.model(inputs)
    
    def loss_func(self, probs, labels):
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
        return tf.reduce_mean(loss)
    
    def total_accuracy(self, probs, labels):
        acc = 0
        top_five = np.argsort(probs, axis=1)
        top_five = np.array(top_five).reshape((self.batch_size, self.num_classes))
        top_five = top_five[:, -1:]

        for i in range(len(labels)):
            if labels[i] not in performance_dict:
                performance_dict[labels[i]] = 0
            if labels[i] in top_five[i]:
                acc += 1
                performance_dict[labels[i]] += 1
            else:
                performance_dict[labels[i]] -= 1
        return (acc / float(self.batch_size))
    


def train(model, train_inputs, train_labels):
    avg_loss = 0
    num_batches = len(train_inputs) // model.batch_size
    for i in range(num_batches):
        with tf.GradientTape() as tape:
            temp_inputs = train_inputs[i*model.batch_size:(i+1)*model.batch_size]
            temp_train_labels=train_labels[i*model.batch_size:(i+1)*model.batch_size]

            predictions = model.call(temp_inputs)
            loss = model.loss_func(predictions, temp_train_labels)
            avg_loss += loss
            if i % 1000 == 0:
                print("---Batch", i, " Loss: ", loss)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    print("****AVERAGE LOSS: ", avg_loss / float(num_batches))

def test(model, test_inputs, test_labels):
    num_batches = len(test_inputs) // (model.batch_size)

    acc = 0
    for i in range(num_batches):
        batch_inputs = test_inputs[i * model.batch_size:(i+1) * model.batch_size]
        batch_labels = test_labels[i * model.batch_size:(i+1) * model.batch_size]

        batch_inputs = np.array(batch_inputs)
        batch_labels = np.array(batch_labels)

        predictions = model.call(batch_inputs)

        batch_accuracy = model.total_accuracy(predictions, batch_labels)

        if i % 100 == 0:
            print("batch accuracy ", batch_accuracy)
        acc += batch_accuracy

    avg_acc = acc / float(num_batches)
    return avg_acc

def test_single_img(model, img):
    crops = []

    # im not sure of the reasoning behind altering and cropping an image for using the model
    image = alter_image(img)
    image = resize_image(image, 105)
    cropped_images = generate_crop(image, 105, 10)

    for c in cropped_images:
        crops.append(c)

    predictions = model.call(crops)
    print(predictions.shape)
    top_5 = model.get_top_five(predictions)
    print(top_5)

def main():
    print("main")
    model = DeepFont()
    model.load_weights('weights_leaky_relu.h5', by_name=True)

    checkpoint_dir = './checkpoints_df'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(model = model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    if args.restore:
        print("Running test mode")
        checkpoint.restore(manager.latest_checkpoint)
    try:
        with tf.device('/device:' + 'CPU:0'):
            if True:
                imgs = []
                labels = []
                for i in range(250):
                    row = ds[i]
                    labels.extend([row["label"]] * 5)
                    imgs.extend(get_samples(row["image"], 5))
                dataX = np.array(imgs) / 255.0
                dataY = np.array(labels)

                train_inputs, train_labels = train_test_split(dataX, dataY, test_size=0.1, random_state=1)
                for epoch in range(args.num_epochs):
                    print(f'----EPOCH {epoch}----')
                    train(model, train_inputs, train_labels)
                    manager.save()
    except RuntimeError as e:
        print(e)

if __name__ == '__main__':
    main()


