import re
import glob
import random
import math
import csv
import os
import ast

import ndjson
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from custom_logger import logger

MAIN_SEED = 0
SIMPLIFIED_DATA_IMAGE_SIZE = 256
REDUCED_DATA_IMAGE_SIZE = 28
NUMBER_IMAGE_OF_CHANNELS = 1

NPY_FILE_REXEXP = re.compile(r"data/class_(?P<drawing_name>.*)_\d+x\d+_id_(?P<key_id>\d+)_(?P<countrycode>\D+)_r_(?P<recognized>\d).npy")

random.seed(MAIN_SEED)


def get_data_files(file_type="ndjson"):

    if file_type == "ndjson":
        data_files = glob.glob("data/*ndjson")
    elif file_type == "csv":
        data_files = glob.glob("data/*csv")
    else:
        assert False, "Wrong type!"
    number_of_classes = len(data_files)
    return data_files, number_of_classes

def get_numpy_drawings_list(reduced_set=None):
    numpy_drawings_list = glob.glob("data/*.npy")

    if reduced_set is not None:
        numpy_drawings_list = numpy_drawings_list[1:reduced_set]

    return numpy_drawings_list


def get_line(start, end):
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end

    >>> points1 = get_line((0, 0), (3, 4))
    >>> points2 = get_line((3, 4), (0, 0))
    >>> assert(set(points1) == set(points2))
    >>> print points1
    [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
    >>> print points2
    [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
    """
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points


def convert_list_image_to_numpy_array(ndjson_drawing):
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

            points = get_line(start, end)
            logger.debug("points: " + str(points))

            for p in points:
                np_drawing[p[0]][p[1]] = 1

    # plt.matshow(np_drawing)
    # plt.show()

    return np_drawing


def convert_images_from_ndjson_file_into_numpy_arrays_and_save(data_file,
                                                               drawing_name,
                                                               file_extension,
                                                               drawings_per_file):

    with open(data_file) as f:
        if file_extension == ".ndjson":
            data = ndjson.load(f)
        elif file_extension == ".csv":
            reader = csv.reader(f)
            data = list(reader)
            logger.info("data: {0}".format(data[0]))
            data = data[1:]
        else:
            assert False, "Wrong file extension!"

    n_drawings = len(data)
    logger.info("Number of drawings = {0}".format(n_drawings))

    if drawings_per_file == None:
        drawings_per_file = n_drawings

    for i in range(drawings_per_file):
        if i % 1000 == 0:
            logger.info("Processing: {0}/{1}".format(i, n_drawings))

        if file_extension == ".ndjson":
            current_drawing = data[i]["drawing"]
            recognized = data[i]["recognized"]
            countrycode = data[i]["countrycode"]
            key_id = data[i]["key_id"]
        elif file_extension == ".csv":
            countrycode = data[i][0] # countrycode
            # drawing is a string that represents a list so we need to
            # evaluate it to have a normal list.
            current_drawing = ast.literal_eval(data[i][1]) # drawing
            key_id = data[i][2] # key_id
            # timestamp is not necessary
            recognized = data[i][3] # recognized
            word = data[i][5].replace(" ", "_") # word
            logger.debug("word: {0} drawing_name: {1}".format(word, drawing_name))
            assert word == drawing_name, "Word and drawing name do not match!"
        else:
            assert False, "Wrong file extension!"

        # logger.info("ndjson_drawing: {0}".format(current_drawing))
        np_drawing = convert_list_image_to_numpy_array(current_drawing)

        np_drawing = cv.resize(np_drawing,
                              (REDUCED_DATA_IMAGE_SIZE, REDUCED_DATA_IMAGE_SIZE),
                              interpolation = cv.INTER_AREA)

        # logger.info("np_drawing after resize: {0}".format(np_drawing))

        if recognized == "true":
            recognized = "1"
        else:
            recognized = "0"

        output_file_name = "data/class_{0}_{1}x{2}_id_{3}_{4}_r_{5}.npy".format(drawing_name,
                                                                                REDUCED_DATA_IMAGE_SIZE,
                                                                                REDUCED_DATA_IMAGE_SIZE,
                                                                                key_id,
                                                                                countrycode,
                                                                                recognized)

        # logger.info("output_file_name: {0}".format(output_file_name))
        np.save(output_file_name, np_drawing)


def convert_ndjson_simplified_data_into_numpy_arrays(ndjson_csv_file_list,
                                                     drawings_per_file=None):

    _, file_extension = os.path.splitext(ndjson_csv_file_list[0])

    n_files = len(ndjson_csv_file_list)

    for i in range(n_files):

        if file_extension == ".ndjson":
            rc = re.compile(r"data/full%2Fsimplified%2F(?P<drawing_name>.*).ndjson")
        elif file_extension == ".csv":
            rc = re.compile(r"data/(?P<drawing_name>.*).csv")
        else:
            assert False, "Wrong file extension!"

        rm = rc.match(ndjson_csv_file_list[i])
        logger.info("ndjson_file: {0}".format(ndjson_csv_file_list[i]))
        logger.info("rm status: {0}".format(rm))
        logger.info("found drawing_name: {0}".format(rm.group("drawing_name")))

        drawing_name = rm.group("drawing_name").replace(" ", "_")

        convert_images_from_ndjson_file_into_numpy_arrays_and_save(ndjson_csv_file_list[i],
                                                                   drawing_name,
                                                                   file_extension,
                                                                   drawings_per_file)

def get_labels(numpy_drawings_list):

    n = len(numpy_drawings_list)

    count_labels = {}
    labels = [None for i in range(n)]
    for i in range(n):
            rm = NPY_FILE_REXEXP.match(numpy_drawings_list[i])
            assert rm, "Regexp not matched with: {0}!".format(numpy_drawings_list[i])

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

def split_the_numpy_drawings_into_test_train_evaluate_datasets(reduced_set=None):

    numpy_drawings_list = get_numpy_drawings_list(reduced_set=reduced_set)
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

    #logger.debug(le.transform(["axe", "bat", "baseball_bat"]))

    #logger.debug(labels)

    labels = le.transform(labels)
    logger.debug(labels)

    x_train, x_test, y_train, y_test = train_test_split(numpy_drawings_list,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=MAIN_SEED)

    logger.info(" --- Size of split data --- ")
    logger.info("x_train length: {0}".format(len(x_train)))
    logger.info("y_train length: {0}".format(len(y_train)))

    logger.info("x_test length: {0}".format(len(x_test)))
    logger.info("y_test length: {0}".format(len(y_test)))
    logger.info(" ---                    --- ")

    return x_train, y_train, x_test, y_test, le

def predict_single_image(npy_drawing_file, model, le):
    logger.info(" --- --- --- --- --- --- --- --- --- --- ")
    logger.info("Making a prediction for: {0}".format(npy_drawing_file))

    rm = NPY_FILE_REXEXP.match(npy_drawing_file)
    assert rm, "Regexp not matched!"
    l = rm.group("drawing_name")
    logger.info("We have a: {0}".format(l))

    x = np.load(npy_drawing_file).reshape((1, REDUCED_DATA_IMAGE_SIZE, REDUCED_DATA_IMAGE_SIZE, 1))
    p = model.predict(x)
    # logger.info("Model prediction p: {0}".format(p))

    p_max = np.unravel_index(np.argmax(p[0], axis=None), p[0].shape)

    logger.info("p has maximum at: {0}".format(p_max))
    p_label = le.inverse_transform(p_max)
    logger.info("The model predicted: {0}".format(p_label))

    plt.matshow(x.reshape((REDUCED_DATA_IMAGE_SIZE, REDUCED_DATA_IMAGE_SIZE)))
    plt.show()
    logger.info(" --- --- --- --- --- --- --- --- --- --- ")

def read_npy_drawing_file_lists_and_return_data_array(x_npy_drawing_file_list,
                                                      y_npy_drawing_labels_list,
                                                      le,
                                                      number_of_classes):

    """
    This function is used mainly to prepare the data for the evaulate/predict
    methods of the model.

    """
    logger.info("Reading data from *npy files and packing them into one big numpy array.")

    n_x = len(x_npy_drawing_file_list)
    n_y = len(y_npy_drawing_labels_list)

    assert n_x == n_y, "x and y dimensions do not match!"

    x_drawings = np.zeros((n_x, REDUCED_DATA_IMAGE_SIZE, REDUCED_DATA_IMAGE_SIZE, NUMBER_IMAGE_OF_CHANNELS))
    y_labels = np.zeros((n_y, number_of_classes))

    for i in range(n_x):
        x = np.load(x_npy_drawing_file_list[i]).reshape((REDUCED_DATA_IMAGE_SIZE, REDUCED_DATA_IMAGE_SIZE, NUMBER_IMAGE_OF_CHANNELS))

        rm = NPY_FILE_REXEXP.match(x_npy_drawing_file_list[i])
        assert rm, "Regexp not matched!"
        l = rm.group("drawing_name")

        label = le.transform([l])
        logger.debug("label: {0}, expected label: {1}".format(label, y_npy_drawing_labels_list[i]))

        assert label == y_npy_drawing_labels_list[i], "Labels do not match!"

        x_drawings[i, :, :, :] = x
        y_labels[i, y_npy_drawing_labels_list[i]] = 1.0

    return x_drawings, y_labels
