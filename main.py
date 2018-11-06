import re
import glob
import random

import ndjson
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from custom_logger import logger
import support_functions as sf

MAIN_SEED = 0
SIMPLIFIED_DATA_IMAGE_SIZE = 256
REDUCED_DATA_IMAGE_SIZE = 28

random.seed(MAIN_SEED)

def get_data_files():
    data_files = glob.glob("data/*ndjson")
    return data_files

def get_numpy_drawings_list():
    numpy_drawings_list = glob.glob("data/*.npy")
    return numpy_drawings_list

def convert_ndjson_image_to_numpy_array(ndjson_drawing):
    np_drawing = np.zeros((SIMPLIFIED_DATA_IMAGE_SIZE, SIMPLIFIED_DATA_IMAGE_SIZE))

    logger.debug("---> ndjson drawing START <---")
    logger.debug(ndjson_drawing)
    logger.debug("---> ndjson drawing END <---")

    for i in range(len(ndjson_drawing)):
        for k in range(1, len(ndjson_drawing[i][0])):
            logger.debug("drawing[i][0][k]: " + str(ndjson_drawing[i][0][k]))
            logger.debug("drawing[i][1][k]: " + str(ndjson_drawing[i][1][k]))

            start = (ndjson_drawing[i][0][k-1], ndjson_drawing[i][1][k-1])
            end = (ndjson_drawing[i][0][k], ndjson_drawing[i][1][k])

            points = sf.get_line(start, end)
            logger.debug("points: " + str(points))

            for p in points:
                np_drawing[p[0]][p[1]] = 1

    # plt.matshow(np_drawing)
    # plt.show()

    return np_drawing


def convert_images_from_ndjson_file_into_numpy_arrays_and_save(ndjson_file):

    with open(ndjson_file) as f:
        data = ndjson.load(f)

    rc = re.compile(r"data/full%2Fsimplified%2F(?P<drawing_name>.*).ndjson")

    rm = rc.match(ndjson_file)
    logger.info("ndjson_file: {0}".format(ndjson_file))
    logger.info("rm status: {0}".format(rm))

    drawing_name = rm.group("drawing_name").replace(" ", "_")

    n_drawings = len(data)
    logger.info("Number of drawings = {0}".format(n_drawings))

    for i in range(n_drawings):
        if i % 1000 == 0:
            logger.info("Processing: {0}/{1}".format(i, n_drawings))

        logger.debug("found drawing_name: {0}".format(rm.group("drawing_name")))
        ndjson_drawing = data[i]["drawing"]

        np_drawing = convert_ndjson_image_to_numpy_array(ndjson_drawing)

        np_drawing = cv.resize(np_drawing,
                              (REDUCED_DATA_IMAGE_SIZE, REDUCED_DATA_IMAGE_SIZE),
                              interpolation = cv.INTER_AREA)

        # plt.matshow(np_drawing)
        # plt.show()

        output_file_name = "data/class_{0}_{1}x{2}_id_{3}.npy".format(drawing_name,
                                                                             REDUCED_DATA_IMAGE_SIZE,
                                                                             REDUCED_DATA_IMAGE_SIZE,
                                                                             i)
        np.save(output_file_name, np_drawing)


def convert_ndjson_simplified_data_into_numpy_arrays(ndjson_file_list):
    n_files = len(ndjson_file_list)
    for i in range(n_files):
        convert_images_from_ndjson_file_into_numpy_arrays_and_save(ndjson_file_list[i])

def get_labels(numpy_drawings_list):

    rc = re.compile(r"data/class_(?P<drawing_name>.*)_\d+x\d+_id_\d+.npy")
    n = len(numpy_drawings_list)

    count_labels = {}
    labels = [None for i in range(n)]
    for i in range(n):
            rm = rc.match(numpy_drawings_list[i])
            assert rm, "Regexp not matched!"

            l = rm.group("drawing_name")
            labels[i] = l

            if l in count_labels:
                count_labels[l] = count_labels[l] + 1
            else:
                count_labels[l] = 0

    logger.info("---   Histogram of labels   ---")
    for key, value in sorted(count_labels.items()):
        logger.info("{0:10s}: {1}".format(key, value))
    logger.info("---   -------------------   ---")

    return labels

def split_the_numpy_drawings_into_test_train_evaluate_datasets():

    numpy_drawings_list = get_numpy_drawings_list()
    logger.debug("numpy_drawings_list length: {0}".format(len(numpy_drawings_list)))
    logger.debug("Before shuffle")
    logger.debug(numpy_drawings_list)

    random.shuffle(numpy_drawings_list)
    logger.debug("After shuffle")
    logger.debug(numpy_drawings_list)

    labels = get_labels(numpy_drawings_list)

    for i in range(len(labels)):
        logger.debug("{0}  -  {1}".format(numpy_drawings_list[i], labels[i]))

    le = LabelEncoder()
    le.fit_transform(labels)

    logger.debug(le.transform(["axe", "bat", "baseball_bat"]))

    logger.debug(labels)

    labels = le.transform(labels)
    logger.debug(labels)

    x_train, x_test, y_train, y_test = train_test_split(numpy_drawings_list,
                                                        labels,
                                                        test_size=0.33,
                                                        random_state=MAIN_SEED)

    logger.debug("x_train: {0}".format(x_train))
    logger.debug("y_train: {0}".format(y_train))

    logger.debug("x_test: {0}".format(x_test))
    logger.debug("y_test: {0}".format(y_test))


if __name__ == "__main__":
    print("Starting...")

    ndjson_file_list = get_data_files()

    # split_the_numpy_drawings_into_test_train_evaluate_datasets()

    # Open each ndjson file, convert the drawings in it into numpy arrays
    # and save them a *.npy files for further processing.
    convert_ndjson_simplified_data_into_numpy_arrays(ndjson_file_list)
