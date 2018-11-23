import math

import h5py
import numpy as np

from custom_logger import logger
import support_functions as sf

from base_model import SimpleCNN
from base_model import ComplexCNN

if __name__ == "__main__":
    print("Starting...")

    ndjson_file_list, number_of_classes = sf.get_data_files(file_type="csv")
    logger.info("Number of classes: {0}".format(number_of_classes))

    npy_drawing_files = sf.get_numpy_drawings_list(reduced_set=None)

    # x_1 = np.load(npy_drawing_files[0]).reshape((1, sf.REDUCED_DATA_IMAGE_SIZE, sf.REDUCED_DATA_IMAGE_SIZE, 1))
    # x_2 = np.load(npy_drawing_files[1]).reshape((1, REDUCED_DATA_IMAGE_SIZE, REDUCED_DATA_IMAGE_SIZE, 1))

    # x = np.concatenate((x_1, x_2), axis=0)
    # x=x_1
    # logger.info("x.shape: {0}".format(x.shape))

    sn = SimpleCNN(input_shape=(sf.REDUCED_DATA_IMAGE_SIZE, sf.REDUCED_DATA_IMAGE_SIZE), n_classes=number_of_classes)
    #sn = ComplexCNN(input_shape=(sf.REDUCED_DATA_IMAGE_SIZE, sf.REDUCED_DATA_IMAGE_SIZE), n_classes=number_of_classes)

    x_train_file_list, y_train_labels_list, x_test_file_list, y_test_labels_list, le = sf.split_the_numpy_drawings_into_test_train_evaluate_datasets(reduced_set=None)

    n_training_samples = len(x_train_file_list)
    n_test_samples = len(x_test_file_list)

    epochs = 20
    batch_size = 100
    samples_per_epoch = math.ceil(n_training_samples/batch_size)
    logger.info("Batch size: {0}".format(batch_size))
    logger.info("Samples per epoch: {0}".format(samples_per_epoch))
    logger.info("Epochs: {0}".format(epochs))

    h5_filename = "train_class_{0}x{0}.h5".format(sf.REDUCED_DATA_IMAGE_SIZE)
    h5_file = h5py.File(h5_filename, "r")
    x = np.array(h5_file.get("train_data/class_bear_28x28_id_6552723749076992_SK_r_0.npy"))
    x = x.reshape((1, sf.REDUCED_DATA_IMAGE_SIZE, sf.REDUCED_DATA_IMAGE_SIZE, 1))

    sn.fit(h5_file=h5_file,
           x_train_file_list=x_train_file_list,
           y_train_labels_list=y_train_labels_list,
           batch_size=10,
           samples_per_epoch=samples_per_epoch,
           epochs=epochs)

    p = sn.predict(x)
    logger.info("p: {0}".format(p))

    h5_file.close()

    # for i in range(10):
    #     sf.predict_single_image(npy_drawing_files[i], sn, le)


    # x_drawings, y_labels = read_npy_drawing_file_lists_and_return_data_array(x_test_file_list,
    #                                                                          y_test_labels_list,
    #                                                                          le,
    #                                                                          number_of_classes)

    # sn.eval(x_test_file_list=x_test_file_list,
    #         y_test_labels_list=y_test_labels_list,
    #         batch_size=10,
    #         steps=samples_per_epoch)

    test_data_list = sf.get_numpy_drawings_list(reduced_set=None,
                                                data_type="test")




    h5_filename = "test_class_{0}x{0}.h5".format(sf.REDUCED_DATA_IMAGE_SIZE)
    h5_file = h5py.File(h5_filename, "r")

    f = open("submission.csv","w")
    f.write("key_id,word\n")
    n_t = len(test_data_list)
    for i in range(n_t):
        if i % 1000 == 0:
            logger.info("{0}/{1}".format(i,n_t))
        # x = np.load(test_data_list[i]).reshape((1, sf.REDUCED_DATA_IMAGE_SIZE, sf.REDUCED_DATA_IMAGE_SIZE, 1))

        x = h5_file.get(test_data_list[i].replace("\n",""))
        x = np.array(x).reshape((1, sf.REDUCED_DATA_IMAGE_SIZE, sf.REDUCED_DATA_IMAGE_SIZE, 1))

        p = sn.predict(x)
        p = p[0]

        indexes = np.argpartition(p,-3)[-3:]

        # logger.info("Processing file: {0} indexes: {1}".format(test_data_list[i], indexes))

        rm = sf.NPY_FILE_REXEXP_TEST.match(test_data_list[i])

        try:
            key_id = rm.group("key_id")
        except AttributeError:
            logger.info("Error is here: {0}".format(test_data_list[i]))


        p0 = le.inverse_transform([indexes[0]])
        p1 = le.inverse_transform([indexes[1]])
        p2 = le.inverse_transform([indexes[2]])

        p_final = "{0} {1} {2}\n".format(p0[0], p1[0], p2[0])
        f.write("{0},{1}".format(key_id,p_final))

    f.close()
    h5_file.close()
