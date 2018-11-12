import re
import glob
import random
import math

import ndjson
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from custom_logger import logger
import support_functions as sf

from base_model import SimpleCNN

MAIN_SEED = 0
SIMPLIFIED_DATA_IMAGE_SIZE = 256
REDUCED_DATA_IMAGE_SIZE = 28
NUMBER_IMAGE_OF_CHANNELS = 1

NPY_FILE_REXEXP = re.compile(r"data/class_(?P<drawing_name>.*)_\d+x\d+_id_\d+.npy")

random.seed(MAIN_SEED)

def get_data_files():
    data_files = glob.glob("data/*ndjson")
    number_of_classes = len(data_files)
    return data_files, number_of_classes

def get_numpy_drawings_list(reduced_set=None):
    numpy_drawings_list = glob.glob("data/*.npy")

    if reduced_set is not None:
        numpy_drawings_list = numpy_drawings_list[1:reduced_set]

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

    n = len(numpy_drawings_list)

    count_labels = {}
    labels = [None for i in range(n)]
    for i in range(n):
            rm = NPY_FILE_REXEXP.match(numpy_drawings_list[i])
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

    logger.debug(le.transform(["axe", "bat", "baseball_bat"]))

    logger.debug(labels)

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
    logger.info("Model prediction p: {0}".format(p))

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

if __name__ == "__main__":
    print("Starting...")

    ndjson_file_list, number_of_classes = get_data_files()
    logger.info("Number of classes: {0}".format(number_of_classes))


    npy_drawing_files = get_numpy_drawings_list(reduced_set=33)

    x_1 = np.load(npy_drawing_files[0]).reshape((1, REDUCED_DATA_IMAGE_SIZE, REDUCED_DATA_IMAGE_SIZE, 1))
    #x_2 = np.load(npy_drawing_files[1]).reshape((1, REDUCED_DATA_IMAGE_SIZE, REDUCED_DATA_IMAGE_SIZE, 1))

    #x = np.concatenate((x_1, x_2), axis=0)
    x=x_1
    logger.info("x.shape: {0}".format(x.shape))

    sn = SimpleCNN(input_shape=(REDUCED_DATA_IMAGE_SIZE, REDUCED_DATA_IMAGE_SIZE), n_classes=number_of_classes)

    x_train_file_list, y_train_labels_list, x_test_file_list, y_test_labels_list, le = split_the_numpy_drawings_into_test_train_evaluate_datasets(reduced_set=None)

    n_training_samples = len(x_train_file_list)
    n_test_samples = len(x_test_file_list)

    epochs = 3
    batch_size = 1000
    samples_per_epoch = math.ceil(n_training_samples/batch_size)
    logger.info("Batch size: {0}".format(batch_size))
    logger.info("Samples per epoch: {0}".format(samples_per_epoch))
    logger.info("Epochs: {0}".format(epochs))

    sn.fit(x_train_file_list=x_train_file_list,
           y_train_labels_list=y_train_labels_list,
           batch_size=10,
           samples_per_epoch=samples_per_epoch,
           epochs=epochs)

    p = sn.predict(x)
    print(p)

    predict_single_image(npy_drawing_files[0], sn, le)
    predict_single_image(npy_drawing_files[1], sn, le)
    predict_single_image(npy_drawing_files[2], sn, le)

    # x_drawings, y_labels = read_npy_drawing_file_lists_and_return_data_array(x_test_file_list,
    #                                                                          y_test_labels_list,
    #                                                                          le,
    #                                                                          number_of_classes)

    sn.eval(x_test_file_list=x_test_file_list,
            y_test_labels_list=y_test_labels_list,
            batch_size=10,
            steps=samples_per_epoch)

    # split_the_numpy_drawings_into_test_train_evaluate_datasets()

    # Open each ndjson file, convert the drawings in it into numpy arrays
    # and save them a *.npy files for further processing.
    # convert_ndjson_simplified_data_into_numpy_arrays(ndjson_file_list)
