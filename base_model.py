import random
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
config.log_device_placement = True
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
                                  x_train_file_list,
                                  y_train_file_list,
                                  batch_size):

        n_x = len(x_train_file_list)
        n_y = len(y_train_file_list)

        assert n_x == n_y, "x and y dimensions do not match!"
        assert batch_size <= n_x, "Batch size is greater than x_train_file_list!"

        x_y_train = [(x_train_file_list[i], y_train_file_list[i]) for i in range(n_x)]

        while True:

            batch_x = np.zeros((batch_size, self.n_rows, self.n_cols, self.n_channels))
            batch_y = np.zeros((batch_size, self.n_classes))

            random.shuffle(x_y_train)
            for i in range(1, batch_size):
                file = x_y_train[i][0]
                y_val = x_y_train[i][1]

                # logger.info("y_val: {0}".format(y_val))

                x = np.load(file).reshape((self.n_rows, self.n_cols, self.n_channels))

                batch_x[i, :, :, :] = x
                batch_y[i, y_val] = 1.0

                # logger.info("batch_y[i, y_val]: {0}".format(batch_y[i, :]))

            yield batch_x, batch_y


    def fit(self, batch_size, x_train_file_list, y_train_file_list):
        # self.model.fit(x_train, y_train,
        #           batch_size=batch_size,
        #           epochs=epochs,
        #           verbose=1,
        #           validation_data=(x_test, y_test))

        self.model.fit_generator(self._generate_data_from_files(x_train_file_list,
                                                                y_train_file_list,
                                                                batch_size),
                                 samples_per_epoch=50,
                                 nb_epoch=10)


    def predict(self, x):
        prediction = self.model.predict(x)
        return prediction


    def eval(self, x_test, y_test):
        score = self.model.evaluate(x_test, y_test, verbose=0)
        return score
