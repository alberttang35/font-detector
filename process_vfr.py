import numpy as np
from PIL import Image
import cv2 as cv
# from imutils import paths
# import itertools
from datasets import load_dataset
import random
from functools import reduce
import json


def noise_image(img):
    '''
    img: matlike
    '''
    # img_array = np.asarray(img)
    mean = 0.0
    std = 3
    noisy_img = img + np.random.normal(mean, std, img.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255)
    # noise_img = Image.fromarray(np.uint8(noisy_img_clipped))
    return noisy_img_clipped


def blur_image(img): # pil_img
    '''
    img: matlike
    '''
    # img_array = np.asarray(img)
    blur_img = cv.GaussianBlur(img, ksize=(3, 3), sigmaX=random.uniform(2.5, 3.5))
    return blur_img


def affine_rotation(img):
    '''
    img: matlike
    '''
    r, c = img.shape

    point1 = np.float32([[10, 10], [30, 10], [10, 30]])
    point2 = np.float32([[20, 15], [40, 10], [20, 40]])

    A = cv.getAffineTransform(point1, point2)

    output = cv.warpAffine(img, A, (c, r))
    # affine_img = Image.fromarray(np.uint8(output))
    # affine_img = affine_img.resize((105, 105))
    # return affine_img
    return output

def gradient_fill(img):
    '''
    img: matlike
    '''
    laplacian = cv.Laplacian(img, cv.CV_64F)
    # laplacian = cv.resize(laplacian, (105, 105))
    return laplacian

def generate_crop(img, dim, count):
    random.seed(1)
    cropped_images = []
    width = len(np.array(img)[1])

    if width > dim + count:
        bounds = random.sample(range(0, width - dim), count)
        for i in range(count):
            new_img = img.crop((bounds[i], 0, bounds[i] + dim, dim))
            new_img = np.array(new_img) / 255.0 # not sure what this line is for

            cropped_images.append(np.array(new_img))
    return cropped_images

def get_samples(img, count=1):
    image = alter_image(img)
    cv.imshow("altered", np.asarray(image))
    cv.waitKey(0)
    image = resize_image(image, 105)
    # image = image.resize((105, 105))
    print(image.size)
    cv.imshow("resized", np.asarray(image))
    cv.waitKey(0)
    image = generate_crop(image, 105, count) # taking multiple crops from each img seems to work well
    # here what the font multiplies each val by 255 and saves every image
    # multiplying by 255 might be to offset dividing by 255 earlier? but still not sure why go thru that in the first place
    # print(type(image))
    image = np.asarray(image)
    return image

def resize_image(img, dim):
    base_height = dim
    height_percent = base_height/float(img.size[1])
    wsize = int((float(img.size[0])*float(height_percent)))
    img = img.resize((wsize, base_height), Image.Resampling.LANCZOS)
    return img


def alter_image(img):
    img = img.convert("L")
    img_array = np.array(img)

    img = noise_image(img_array)

    img = blur_image(img)

    # from WhatTheFont
    rotation_angle = [-4, -2, 0, 2, 4]
    translate_x = [-5, -3, 0, 3, 5]
    translate_y = [-5, -3, 0, 3, 5]
    angle = random.choice(rotation_angle)
    tx = random.choice(translate_x)
    ty = random.choice(translate_y)
    rows, cols = img.shape
    M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
    M_rotate = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    affined_image = cv.warpAffine(img, M_translate, (cols, rows))
    affined_image = cv.warpAffine(affined_image, M_rotate, (cols, rows))

    affined_image = np.array(affined_image) * random.uniform(0.2, 1.5)
    affined_image = np.clip(affined_image, 0, 255).astype(np.uint8)
    final = Image.fromarray(affined_image)

    return final
    # return affined_image


def get_split(dataset, count):
    ds = load_dataset(dataset, split="train")
    rows = ds[:count]
    # out = reduce(lambda acc, elt: acc.extend(get_samples(elt)) , images, [])
    imageCol = rows["image"]
    labelCol = rows["label"]
    imgs = []
    labels = []
    # print(rows)
    for image, label in zip(imageCol, labelCol):
        # print(row)
        # print(image.size)
        samples = get_samples(image, 4)
        imgs.extend(samples)
        labels.extend([label] * len(samples))
    return imgs, labels

def remove_font(font):
    with open('150_fonts.json', 'r') as f:
        font_list = json.load(f)
    new = {}
    index = 0
    for key in list(font_list.keys()):
        if key == font:
            continue
        new[key] = index
        index += 1
    with open("149_fonts.json", "w") as f:
        json.dump(new, f)

def reverse_dict(filepath):
    f = open(filepath, 'r')
    with open(filepath, 'r') as f:
        content = json.load(f)
    # content = f.read().split()
    reversed = {}
    count = 0
    for line in list(content.keys()):
        reversed[str(count)] = line
        count += 1
    with open('149_fonts_backwards.json', 'w') as f:
        json.dump(reversed, f, indent=4)

    

def main():
    print("main")
    # remove_font("ExPontoPro")
    reverse_dict("149_fonts.json")
    # imgs, _ = get_split("gaborcselle/font-examples", 2)
    # for img in imgs:
    #     cv.imshow("crop", img)
    #     cv.waitKey(0)



if __name__ == '__main__':
    main()