import random
import time
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

from custom_logger import logger

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = False
sess = tf.Session(config=config)
set_session(sess)

class BaseModel(ABC):

    def __init__(self,
                 input_size,
                 n_classes):
        pass

    @abstractmethod
    def fit(self, batch_size, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def eval(self, x, y):
        pass

class SimpleCNN(BaseModel):

    def __init__(self,
                 input_shape,
                 n_classes,
                 n_channels=1):

        self.n_rows = input_shape[0]
        self.n_cols = input_shape[1]
        self.n_channels = n_channels
        self.n_classes = n_classes

        logger.info("imput_shape: {0}".format(input_shape))

        self.inputs = Input(shape=(self.n_rows, self.n_cols, n_channels))

        x = Conv2D(32,
                   kernel_size=(3, 3),
                   activation='relu',
                   data_format="channels_last",
                   name="first_layer")(self.inputs)

        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        self.outputs = Dense(n_classes, activation='softmax')(x)

        self.model = Model(inputs=[self.inputs], outputs=[self.outputs])

        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adadelta(),
                           metrics=['accuracy'])

    def _generate_data_from_files(self,
                                  x_data_file_list,
                                  y_data_labels_list,
                                  batch_size):

        n_x = len(x_data_file_list)
        n_y = len(y_data_labels_list)

        assert n_x == n_y, "x and y dimensions do not match!"
        assert batch_size <= n_x, "Batch size is greater than length of x_data_file_list!"

        x_y_train = [(x_data_file_list[i], y_data_labels_list[i]) for i in range(n_x)]

        start = 0
        stop = batch_size
        while_index = 1

        while True:

            logger.debug("\nIn while loop..., while_index: {0}".format(while_index))
            logger.debug("start: {0} stop: {1}".format(start, stop))

            number_of_elements_in_batch = stop - start
            logger.debug("Number of elements in batch: {0}".format(number_of_elements_in_batch))

            batch_x = np.zeros((number_of_elements_in_batch, self.n_rows, self.n_cols, self.n_channels))
            batch_y = np.zeros((number_of_elements_in_batch, self.n_classes))

            while_index = while_index + 1
            index = 0
            # random.shuffle(x_y_train)
            # logger.info("befor loop - start: {0} stop: {1} n_x: {2}".format(start, stop, n_x))
            for i in range(start, stop):
                file = x_y_train[i][0]
                y_val = x_y_train[i][1]

                # logger.info("i: {0} index: {1} Adding {2} file to batch.".format(i, index, file))
                # logger.info("y_val: {0}".format(y_val))

                x = np.load(file).reshape((self.n_rows, self.n_cols, self.n_channels))

                # It would be good to extract the label from "file" and compare
                # it with y_val but for performance reasons we avoid this.

                batch_x[index, :, :, :] = x
                batch_y[index, y_val] = 1.0
                index = index + 1

                # logger.debug("batch_y[i, y_val]: {0}".format(batch_y[i, :]))

            start = start + batch_size
            stop = stop + batch_size
            # logger.info("after loop - start: {0} stop: {1} n_x: {2}".format(start, stop, n_x))

            if (start > n_x or stop > n_x) or (start == stop):
                start = 0
                stop = batch_size

            if stop > n_x:
                stop = n_x

            # time.sleep(10)

            # logger.info("batch_x: {0} batch_y: {1}".format(len(batch_x),len(batch_y)))

            yield batch_x, batch_y


    def fit(self,
            x_train_file_list,
            y_train_labels_list,
            batch_size,
            samples_per_epoch,
            epochs):
        # self.model.fit(x_train, y_train,
        #           batch_size=batch_size,
        #           epochs=epochs,
        #           verbose=1,
        #           validation_data=(x_test, y_test))

        self.model.fit_generator(self._generate_data_from_files(x_data_file_list=x_train_file_list,
                                                                y_data_labels_list=y_train_labels_list,
                                                                batch_size=batch_size),
                                 samples_per_epoch=samples_per_epoch,
                                 epochs=epochs)


    def predict(self, x):
        prediction = self.model.predict(x)
        return prediction


    def eval(self,
             x_test_file_list,
             y_test_labels_list,
             batch_size,
             steps):

        # score = self.model.evaluate(x_test, y_test, verbose=0)
        # logger.info("Your model scored: {0}".format(score))
        # return score

        score = self.model.evaluate_generator(self._generate_data_from_files(x_data_file_list=x_test_file_list,
                                                                             y_data_labels_list=y_test_labels_list,
                                                                             batch_size=batch_size),
                                              steps=steps)

        logger.info("Your model scored: {0}".format(score))
