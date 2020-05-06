from keras.layers import Conv2D, Flatten, MaxPool2D
from keras.layers import Dropout, Dense
from keras.models import Sequential
from keras.optimizers import SGD


def CNN(input_shape, class_number):
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=input_shape))
    # 16 filters, 3x3 sploty (convolusion), padding = same pilnuje zeby wymiary były takie same
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(class_number, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model


def CNN_article_version(input_shape, class_number):
    """
    NIE DZIAŁA, NIE DOTYKAĆ.
    DON'T TOUCH IT.
    """
    model = Sequential()
    model.add(Conv2D(16, (7, 7), activation='relu', strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (5, 5), activation='relu', strides=(2, 2), padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(2, 2), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(2, 2), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(2, 2), padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(2, 2), padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(class_number, activation='softmax'))
    opt = SGD(lr=0.001, decay=0.1)

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

    return model

