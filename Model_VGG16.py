from keras.layers import MaxPool2D
from keras.models import Sequential
from keras.layers import Conv2D, Dense
import numpy as np
from Evaluation import evaluation


def VGG_16(Train_Data, num_of_class=None):
    model = Sequential()
    model.add(
        Conv2D(input_shape=(Train_Data.shape[1], 1, 1), filters=64, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=1, strides=1))
    model.add(Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=1, strides=1))
    model.add(Conv2D(filters=256, kernel_size=2, padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=2, padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=2, padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=1, strides=1))
    model.add(Conv2D(filters=512, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=1, strides=1))
    model.add(Conv2D(filters=512, kernel_size=2, padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=3, padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=1, strides=1))
    model.add(Dense(units=num_of_class, activation="relu"))
    return model


def Model_VGG16(train_data, train_target, test_data, test_target, Epoch):
    Feat1 = train_data.reshape((train_data.shape[0], train_data.shape[1], 1, 1))
    Feat2 = test_data.reshape((test_data.shape[0], test_data.shape[1], 1, 1))

    model = VGG_16(Feat1, num_of_class=train_target.shape[1])
    model.summary()
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.fit(x=Feat1, y=train_target, epochs=Epoch)
    pred = model.predict(Feat2)
    limit = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= limit] = 1
    pred[pred < limit] = 0
    Eval = evaluation(test_target, pred)
    return Eval, pred

