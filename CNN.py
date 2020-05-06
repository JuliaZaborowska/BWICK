from keras.layers import Conv2D, Flatten
from keras.layers import Dropout, Dense
from keras.models import Sequential
from keras.optimizers import SGD


def CNN(input_shape, class_number):

    model = Sequential()
    model.add(Conv2D(16, (7, 7), activation='relu', strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (5, 5), activation='relu', strides=(2, 2), padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(2, 2), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(2, 2), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(class_number, activation='softmax'))
    opt = SGD(lr=0.001, decay=0.1)

    model.summary()
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy, mse'])

    return model
