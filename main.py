import math
import numpy as np

from custom_logger import logger
import support_functions as sf

from base_model import SimpleCNN


if __name__ == "__main__":
    print("Starting...")

    ndjson_file_list, number_of_classes = sf.get_data_files(file_type="csv")
    logger.info("Number of classes: {0}".format(number_of_classes))

    npy_drawing_files = sf.get_numpy_drawings_list(reduced_set=None)

    x_1 = np.load(npy_drawing_files[0]).reshape((1, sf.REDUCED_DATA_IMAGE_SIZE, sf.REDUCED_DATA_IMAGE_SIZE, 1))
    #x_2 = np.load(npy_drawing_files[1]).reshape((1, REDUCED_DATA_IMAGE_SIZE, REDUCED_DATA_IMAGE_SIZE, 1))

    #x = np.concatenate((x_1, x_2), axis=0)
    x=x_1
    logger.info("x.shape: {0}".format(x.shape))

    sn = SimpleCNN(input_shape=(sf.REDUCED_DATA_IMAGE_SIZE, sf.REDUCED_DATA_IMAGE_SIZE), n_classes=number_of_classes)

    x_train_file_list, y_train_labels_list, x_test_file_list, y_test_labels_list, le = sf.split_the_numpy_drawings_into_test_train_evaluate_datasets(reduced_set=54)

    n_training_samples = len(x_train_file_list)
    n_test_samples = len(x_test_file_list)

    epochs = 50
    batch_size = 5
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

    for i in range(10):
        sf.predict_single_image(npy_drawing_files[i], sn, le)


    # x_drawings, y_labels = read_npy_drawing_file_lists_and_return_data_array(x_test_file_list,
    #                                                                          y_test_labels_list,
    #                                                                          le,
    #                                                                          number_of_classes)

    sn.eval(x_test_file_list=x_test_file_list,
            y_test_labels_list=y_test_labels_list,
            batch_size=10,
            steps=samples_per_epoch)
