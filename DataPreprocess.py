import os
import sys
import numpy as np
#import cPickle as pickle
import pickle
from natsort import natsorted
from PIL import Image
import cv2
import matplotlib.pyplot as plt

DIR_TRAIN_DATA_IMG = "imageTrain10"
FILE_TRAIN_DATA_OUTPUT="train10.csv"
DIR_TEST_DATA_IMG="imageTest10"
IMAGE_SIZE = 128
IMAGE_NUM_CHANNELS = 1
CROP_SIZE = 200
NUM_LABELS = 37




def image_process(img_array):

    img_array = cv2.GaussianBlur(img_array, (17, 17), 0)
    img_array = cv2.bitwise_not(img_array)
    image=Image.fromarray(img_array)
    if CROP_SIZE > 0:
        (width, height) = image.size
        left = (width-CROP_SIZE) / 2
        top = (height-CROP_SIZE) / 2
        right = left + CROP_SIZE
        bottom = top + CROP_SIZE
        image = image.crop((left, top, right, bottom))
    if not (IMAGE_SIZE, IMAGE_SIZE) == image.size:
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)

    image = np.array(image)
    #image = np.rollaxis(image, 2)
    image = image.reshape(-1)
    return image

def createTestingDataSet():

    test_current_batch = np.zeros((IMAGE_SIZE * IMAGE_SIZE * IMAGE_NUM_CHANNELS, 0), dtype=np.uint8)
    test_names = [d for d in os.listdir(DIR_TEST_DATA_IMG) if d.endswith('.jpg')]
    for image_name in test_names:
        image_file_name = image_name
        image = cv2.imread(os.path.join(DIR_TEST_DATA_IMG, image_file_name), cv2.IMREAD_GRAYSCALE)
        try:
            image = image_process(image)
        except ValueError:
            print ("problem with test image {}".format(image_file_name))
            sys.exit(1)
        image = image.reshape(-1, 1)
        test_current_batch = np.hstack((test_current_batch, image))

    test_image = test_current_batch.T

    pickle_out = open("./pickles/X_test.pickle", "wb")
    pickle.dump(test_image, pickle_out)
    pickle_out.close()


def createTrainingDataSet():
    label_data = np.genfromtxt(FILE_TRAIN_DATA_OUTPUT, dtype=np.float32, delimiter=',', skip_header=1)
    train_current_batch = np.zeros((IMAGE_SIZE * IMAGE_SIZE * IMAGE_NUM_CHANNELS, 0), dtype=np.uint8)
    train_current_batch_label = np.zeros((NUM_LABELS, 0), dtype=np.float32)

    sorted_idx = label_data[:, 0].argsort()
    label_data = label_data[sorted_idx, 1:]
    label_data = label_data.T
    train_names = [d for d in os.listdir(DIR_TRAIN_DATA_IMG) if d.endswith('.jpg')]
    train_names = natsorted(train_names)
    num_trains = len(train_names)
    data_mean = np.zeros((IMAGE_SIZE * IMAGE_SIZE * IMAGE_NUM_CHANNELS, 1), dtype=np.float32)
    train_order = np.random.permutation(num_trains)
    # Training data set
    for i in train_order:
        image_file_name = train_names[i]
        image = cv2.imread(os.path.join(DIR_TRAIN_DATA_IMG, image_file_name), cv2.IMREAD_GRAYSCALE)
        try:
            image = image_process(image)
        except ValueError:
            print("problem with train image {}".format(image_file_name))
            sys.exit(1)
        image = image.reshape(-1, 1)
        data_mean += image
        train_current_batch = np.hstack((train_current_batch, image))
        train_current_batch_label = np.hstack((train_current_batch_label,
                                               label_data[:, i].reshape(-1, 1)))
    train_image = train_current_batch.T
    train_values = train_current_batch_label.T

    pickle_out = open("./pickles/X.pickle", "wb")
    pickle.dump(train_image, pickle_out)
    pickle_out.close()

    pickle_out = open("./pickles/y.pickle", "wb")
    pickle.dump(train_values, pickle_out)
    pickle_out.close()


def main():
    image = cv2.imread(os.path.join(DIR_TRAIN_DATA_IMG, "100143.jpg"), cv2.IMREAD_GRAYSCALE)
    image_process(image)

    createTrainingDataSet()
    createTestingDataSet()
    pickle_in = open("./pickles/X_test.pickle", "rb")
    X = pickle.load(pickle_in)
    image = Image.fromarray(X[0].reshape(128, 128))
    plt.imshow(image, cmap="gray")
    plt.show()

if __name__ == '__main__':
    main()
