import glob

import ndjson
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from custom_logger import logger
import support_functions as sf

SIMPLIFIED_DATA_IMAGE_SIZE = 256
REDUCED_DATA_IMAGE_SIZE = 28

def get_data_files():
    data_files = glob.glob("data/*ndjson")
    return data_files


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


def convert_images_from_ndjson_file_into_numpy_arrays(ndjson_file):

    with open(ndjson_file) as f:
        data = ndjson.load(f)

    n_drawings = len(data)
    logger.info("Number of drawings = {0}".format(n_drawings))

    np_drawing_array = np.zeros((n_drawings, REDUCED_DATA_IMAGE_SIZE, REDUCED_DATA_IMAGE_SIZE))

    logger.debug("np_drawing_array size: {0}".format(np_drawing_array.shape))
    for i in range(n_drawings):
        logger.info("Processing: {0}/{1}".format(i, n_drawings))
        ndjson_drawing = data[i]["drawing"]

        np_drawing = convert_ndjson_image_to_numpy_array(ndjson_drawing)

        np_drawing = cv.resize(np_drawing,
                               (REDUCED_DATA_IMAGE_SIZE, REDUCED_DATA_IMAGE_SIZE),
                               interpolation = cv.INTER_AREA)

        # plt.matshow(np_drawing)
        # plt.show()

        np_drawing_array[i, :, :] = np_drawing

    return np_drawing_array


def convert_ndjson_simplified_data_into_numpy_arrays(ndjson_file_list):

    # First du,,y element so we have something to concatenate with.
    np_reduced_drawings = np.zeros((1, REDUCED_DATA_IMAGE_SIZE, REDUCED_DATA_IMAGE_SIZE))

    n_files = len(ndjson_file_list)
    for i in range(n_files):

        np_drawing_array = convert_images_from_ndjson_file_into_numpy_arrays(ndjson_file_list[i])
        np_reduced_drawings = np.concatenate((np_reduced_drawings, np_drawing_array), axis=0)

    logger.debug("Shape of np_reduced_drawings before droping dummy element: {0}".format(np_reduced_drawings.shape))

    # Drop the first dummy element.
    np_reduced_drawings = np_reduced_drawings[1:, :, :]
    logger.debug("Shape of np_reduced_drawings after dropping dummy element: {0}".format(np_reduced_drawings.shape))

    return np_reduced_drawings

if __name__ == "__main__":
    print("Starting...")

    ndjson_file_list = get_data_files()

    convert_ndjson_simplified_data_into_numpy_arrays(ndjson_file_list)

    #np_drawing_array = convert_images_from_ndjson_file_into_numpy_arrays(data_files[1])

    # data = None
    # with open("data/full%2Fsimplified%2Faxe.ndjson") as f:
    #     data = ndjson.load(f)
    #
    # ndjson_drawing = data[243]["drawing"]
    # convert_ndjson_image_to_numpy_array(ndjson_drawing)
    #
    #
    # m_drawing = np.zeros((256, 256))
    # print(m_drawing.shape)
    # print(type(data))
    #
    # drawing = data[243]["drawing"]
    #
    # print(drawing)
    # print("len: {0}".format(len(drawing)))
    # # print(np.array(drawing))
    # print(drawing[0])
    # print(len(drawing[0]))
    # print(drawing[0][1])
    # print(len(drawing[0][1]))
    #
    # print(drawing[0][1][0])
    #
    # for i in range(len(drawing)):
    #         for k in range(1, len(drawing[i][0])):
    #             print("drawing[i][0][k]: " + str(drawing[i][0][k]))
    #             print("drawing[i][1][k]: " + str(drawing[i][1][k]))
    #
    #             start = (drawing[i][0][k-1], drawing[i][1][k-1])
    #             end = (drawing[i][0][k], drawing[i][1][k])
    #
    #             points = sf.get_line(start, end)
    #             print("points: " + str(points))
    #
    #             for p in points:
    #                 m_drawing[p[0]][p[1]] = 1
    #
    #
    # plt.matshow(m_drawing)
    # plt.show()