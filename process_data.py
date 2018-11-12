from custom_logger import logger
import support_functions as sf

if __name__ == "__main__":
    print("Starting...")

    ndjson_file_list, number_of_classes = sf.get_data_files()
    logger.info("Number of classes: {0}".format(number_of_classes))


    # Open each ndjson file, convert the drawings in it into numpy arrays
    # and save them a *.npy files for further processing.
    sf.convert_ndjson_simplified_data_into_numpy_arrays(ndjson_file_list)
