from abc import ABC, abstractmethod

import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from custom_logger import logger

class BaseModel(ABC):

    def __init__(self,
                 input_size,
                 n_classes):
        pass

    @abstractmethod
    def fit(self, x, y):
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
                 n_classes):

        rows = input_shape[0]
        cols = input_shape[1]

        logger.info("imput_shape: {0}".format(input_shape))

        self.inputs = Input(shape=(rows, cols, 1))

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


    def fit(self, batch_size, epochs, x_train, y_train, x_test, y_test):
        self.model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))


    def predict(self, x):
        prediction = self.model.predict(x)
        return prediction


    def eval(self, x_test, y_test):
        score = self.model.evaluate(x_test, y_test, verbose=0)
        return score
