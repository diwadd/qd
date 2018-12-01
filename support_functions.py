import re
import glob
import random
import math
import csv
import os
import ast
import time

import ndjson
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from custom_logger import logger

MAIN_SEED = 0
SIMPLIFIED_DATA_IMAGE_SIZE = 256
REDUCED_DATA_IMAGE_SIZE = 128
NUMBER_IMAGE_OF_CHANNELS = 4
PADDING_SZIE = 6

NPY_FILE_REXEXP = re.compile(r"train_data/class_(?P<drawing_name>.*)_\d+x\d+_id_(?P<key_id>\d+)_(?P<countrycode>\D+)_r_(?P<recognized>\d).npy")
NPY_FILE_REXEXP_TEST = re.compile(r"test_data/class_(?P<drawing_name>.*)_\d+x\d+_id_(?P<key_id>\d+)_(?P<countrycode>\D+|)_r_(?P<recognized>\d).npy")

BASE_COLOR = "green"
PROTEIN_COLORS = ["green", "blue", "red", "yellow"]

random.seed(MAIN_SEED)


def get_data_files(file_type="ndjson",
                   data_type="train",
                   color="green"):

    search_string = "../{0}/*_{1}.{2}".format(data_type, color, file_type)

    logger.info("Looking for {0}".format(search_string))
    data_files = glob.glob(search_string)

    return data_files


def get_data_point_ids_and_labels(ids_labels_file):

    with open(ids_labels_file) as f:
        reader = csv.reader(f)
        ids_labels = list(reader)

    return ids_labels[1:]


def convert_ids_labels_into_dict(ids_labels):

    ids_labels_dict = {}
    n = len(ids_labels)
    for i in range(n):

        id = ids_labels[i][0]
        label = ids_labels[i][1]

        label = list(map(int, label.split()))

        if id not in ids_labels_dict:
            ids_labels_dict[id] = label
        else:
            assert False, "Duplicate id! Somethings wrong with the input!"

    return ids_labels_dict


def check_that_all_other_colors_exist(data_file):

    for c in PROTEIN_COLORS:
        # This loop is always redundant over one of the colors.
        ocf = data_file.replace(BASE_COLOR, c)

        logger.debug("File exists: {0}".format(ocf))
        assert os.path.isfile(ocf), "File {0} does not exist".format(ocf)


def generate_four_files(data_file,
                        four_files):

    logger.debug("Generating four files.")
    n = len(PROTEIN_COLORS)
    for i in range(n):
        ocf = data_file.replace(BASE_COLOR, PROTEIN_COLORS[i])
        four_files[i] = ocf


def load_single_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # logger.info("Image shape: {0}".format(img.shape))
    img = cv2.resize(img,
                     (REDUCED_DATA_IMAGE_SIZE, REDUCED_DATA_IMAGE_SIZE),
                     interpolation=cv2.INTER_AREA)

    return img


def pack_images_into_one_npy_array(four_files):

    n = len(four_files)

    four_channel_img = np.zeros((REDUCED_DATA_IMAGE_SIZE,
                                 REDUCED_DATA_IMAGE_SIZE,
                                 NUMBER_IMAGE_OF_CHANNELS))
    for i in range(n):
        img = load_single_image(four_files[i])

        four_channel_img[:, :, i] = img

    return four_channel_img


def pack_images_into_npy_array(data_files, ids_labels_dict=None):

    if ids_labels_dict is not None:
        data_type = "train"
    else:
        data_type = "test"

    n = len(data_files)

    for i in range(n):
        if i % 10 == 0:
            logger.info("We are at: {0}/{1}".format(i, n))

        df = data_files[i]

        logger.debug("--- --- ---")
        check_that_all_other_colors_exist(df)

        four_files = ["", "", "", ""]
        generate_four_files(df, four_files)

        logger.debug("The four generated files:")
        for f in four_files:
            logger.debug(f)

        four_channel_img = pack_images_into_one_npy_array(four_files)
        logger.debug("Shape of four_channel_img: {0}".format(four_channel_img.shape))

        # Get id
        rc = re.compile(r"../{0}/(?P<id>.*)_(blue|red|yellow|green).png".format(data_type))
        rm = rc.match(four_files[0])

        assert rm, "Regexp not matched in function: pack_images_into_npy_array!"

        id = rm.group("id")

        if ids_labels_dict is not None:

            label = list(map(str, ids_labels_dict[id]))
            label = "_".join(label)
        else:
            label = "None"

        logger.debug("Label: {0}".format(label))
        npy_file_name = "../{0}_data/img_{1}_s_{2}x{2}_label_{3}.npy".format(data_type,
                                                                             id,
                                                                             REDUCED_DATA_IMAGE_SIZE,
                                                                             label)

        logger.debug("npy file name: {0}".format(npy_file_name))
        np.save(npy_file_name, four_channel_img)




def get_numpy_drawings_list(reduced_set=None,
                            data_type="train",
                            class_list=None):

    numpy_drawings_list = []

    nl = len(QUICK_DRAW_LABELS)
    logger.info("Number of QUICK_DRAW_LABELS: {0}".format(nl))
    for i in range(nl):

        l = QUICK_DRAW_LABELS[i]
        if data_type == "train":
            class_numpy_drawings_list = glob.glob("train_data/class_{0}_{1}x{1}*.npy".format(l, REDUCED_DATA_IMAGE_SIZE))
        else:
            class_numpy_drawings_list = glob.glob("test_data/class_{0}_{1}x{1}*.npy".format(l, REDUCED_DATA_IMAGE_SIZE))

        if reduced_set is not None:
            class_numpy_drawings_list = class_numpy_drawings_list[0:reduced_set]

        numpy_drawings_list = numpy_drawings_list + class_numpy_drawings_list

    return numpy_drawings_list



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
                                                               drawings_per_file,
                                                               N_DRAWINGS_PER_ARRAY,
                                                               data_type="train"):

    _, file_extension = os.path.splitext(data_file)

    if file_extension == ".ndjson":
        rc = re.compile(r"data/full%2Fsimplified%2F(?P<drawing_name>.*).ndjson")
    elif file_extension == ".csv":
        rc = re.compile(r"data/(?P<drawing_name>.*).csv")
    else:
        assert False, "Wrong file extension!"

    if data_type == "train":
        rm = rc.match(data_file)
        logger.info("ndjson_file: {0}".format(data_file))
        logger.info("rm status: {0}".format(rm))
        logger.info("found drawing_name: {0}".format(rm.group("drawing_name")))
        drawing_name = rm.group("drawing_name").replace(" ", "_")
    else:
        drawing_name = None

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

    a_index = 0
    i_index = 0
    np_drawing_array = np.zeros((N_DRAWINGS_PER_ARRAY, REDUCED_DATA_IMAGE_SIZE, REDUCED_DATA_IMAGE_SIZE))

    n_chunks = int(drawings_per_file/N_DRAWINGS_PER_ARRAY)
    logger.info("Number of chunks: {0}".format(n_chunks))
    last_chunk_empty = True

    for i in range(drawings_per_file):
        if i % 10000 == 0:
            logger.info("Processing: {0}/{1}".format(i, n_drawings))

        if file_extension == ".ndjson":
            current_drawing = data[i]["drawing"]
            recognized = data[i]["recognized"]
            countrycode = data[i]["countrycode"]
            key_id = data[i]["key_id"]
        elif file_extension == ".csv":
            if data_type == "train":
                countrycode = data[i][0] # countrycode
                # drawing is a string that represents a list so we need to
                # evaluate it to have a normal list.
                current_drawing = ast.literal_eval(data[i][1]) # drawing
                key_id = data[i][2] # key_id
                # timestamp is not necessary
                recognized = data[i][3] # recognized
                # word = data[i][5].replace(" ", "_") # word
                # logger.debug("word: {0} drawing_name: {1}".format(word, drawing_name))
                # assert word == drawing_name, "Word and drawing name do not match!"
            elif data_type == "test":
                # The test data set has mixed columns with respect to the train data.
                key_id = data[i][0]
                countrycode = data[i][1]
                current_drawing = ast.literal_eval(data[i][2]) # drawing
                recognized = "0"
            else:
                pass
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


        # if data_type == "train":
        #     output_file_name = "train_data/class_{0}_{1}x{2}_id_{3}_{4}_r_{5}.npy".format(drawing_name,
        #                                                                             REDUCED_DATA_IMAGE_SIZE,
        #                                                                             REDUCED_DATA_IMAGE_SIZE,
        #                                                                             key_id,
        #                                                                             countrycode,
        #                                                                             recognized)
        if data_type == "test":
            output_file_name = "test_data/class_test_{1}x{2}_id_{3}_{4}_r_0.npy".format("test",
                                                                                         REDUCED_DATA_IMAGE_SIZE,
                                                                                         REDUCED_DATA_IMAGE_SIZE,
                                                                                         key_id,
                                                                                         countrycode)
            np.save(output_file_name, np_drawing)
            continue


        if i_index == n_chunks:
            np_drawing = np_drawing.reshape((1, REDUCED_DATA_IMAGE_SIZE, REDUCED_DATA_IMAGE_SIZE))
            np_drawing_array = np.concatenate((np_drawing_array, np_drawing), axis=0)
            last_chunk_empty = False
        else:
            np_drawing_array[a_index, :, :] = np_drawing[:, :]
            a_index = a_index + 1

        if a_index == N_DRAWINGS_PER_ARRAY:
            np_drawinf_array_filename = "train_data/class_{0}_{1}x{2}_id_{3}_XX_r_0.npy".format(drawing_name,
                                                                                                REDUCED_DATA_IMAGE_SIZE,
                                                                                                REDUCED_DATA_IMAGE_SIZE,
                                                                                                str(i_index).zfill(PADDING_SZIE))
            np.save(np_drawinf_array_filename, np_drawing_array)
            a_index = 0
            i_index = i_index + 1

            if i_index == n_chunks:
                logger.info("Processed {0} chunks {1}".format(i_index, n_chunks))
                np_drawing_array = np.zeros((1, REDUCED_DATA_IMAGE_SIZE, REDUCED_DATA_IMAGE_SIZE))
            else:
                np_drawing_array = np.zeros((N_DRAWINGS_PER_ARRAY, REDUCED_DATA_IMAGE_SIZE, REDUCED_DATA_IMAGE_SIZE))

        # logger.info("output_file_name: {0}".format(output_file_name))



    if last_chunk_empty == False:
         np_drawinf_array_filename = "train_data/class_{0}_{1}x{2}_id_{3}_XX_r_0.npy".format(drawing_name,
                                                                                             REDUCED_DATA_IMAGE_SIZE,
                                                                                             REDUCED_DATA_IMAGE_SIZE,
                                                                                             str(i_index).zfill(PADDING_SZIE))
         np.save(np_drawinf_array_filename, np_drawing_array[1:, :, :])


def convert_ndjson_simplified_data_into_numpy_arrays(ndjson_csv_file_list,
                                                     drawings_per_file=100,
                                                     N_DRAWINGS_PER_ARRAY=10):

    n_files = len(ndjson_csv_file_list)

    for i in range(n_files):

        start = time.time()
        logger.info("File number: {0}".format(i))

        convert_images_from_ndjson_file_into_numpy_arrays_and_save(ndjson_csv_file_list[i],
                                                                   drawings_per_file,
                                                                   N_DRAWINGS_PER_ARRAY)

        end = time.time()
        logger.info("Elapsed time: {0} s\n".format(end - start))


def get_labels(numpy_drawings_list):

    n = len(numpy_drawings_list)

    count_labels = {}
    labels_set = set()
    labels = [None for i in range(n)]
    for i in range(n):
            rm = NPY_FILE_REXEXP.match(numpy_drawings_list[i])
            assert rm, "Regexp not matched with: {0}!".format(numpy_drawings_list[i])

            l = rm.group("drawing_name")
            labels[i] = l
            labels_set.add(l)

            if l in count_labels:
                count_labels[l] = count_labels[l] + 1
            else:
                count_labels[l] = 1

    logger.info("---   Histogram of labels   ---")
    logger.info("Number of labels in histogram: {0}".format(len(count_labels)))
    for key, value in sorted(count_labels.items()):
        logger.info("{0:10s}: {1}".format(key, value))
    logger.info("---   -------------------   ---")
    logger.info("Number of labels in set: {0}".format(len(labels_set)))
    logger.info(labels_set)
    logger.info("---   -------------------   ---")

    logger.info("My labels size: {0}".format(len(labels)))

    return labels

def split_the_numpy_drawings_into_test_train_evaluate_datasets(reduced_set=None,
                                                               test_size=0.05):

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


    # logger.info("Our labels: {0}".format(labels))

    le = LabelEncoder()
    le.fit_transform(QUICK_DRAW_LABELS)

    logger.info("Checking the labels mapping")
    logger.info(le.transform(["axe", "bat", "baseball_bat"]))

    labels = le.transform(labels)
    logger.debug(labels)

    x_train, x_test, y_train, y_test = train_test_split(numpy_drawings_list,
                                                        labels,
                                                        test_size=test_size,
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
