import numpy as np
from keras import Sequential
from keras.applications import ResNet50
from keras.layers import Dense
from Evaluation import evaluation


def Model_ResNet(train_data, train_target, test_data, test_target, epoch):

    IMG_SIZE = int(train_data.shape[1] / 2)
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))
    inputs = (Train_X.shape[1], Train_X.shape[2], Train_X.shape[3])

    base_model = Sequential()
    base_model.add(ResNet50(include_top=False, weights='imagenet', pooling='max', input_shape=inputs))
    base_model.add(Dense(units=train_target.shape[1], activation='linear'))
    base_model.compile(loss='binary_crossentropy', metrics=['acc'])
    base_model.summary()
    base_model.fit(Train_X, train_target, epochs=epoch, steps_per_epoch=100)
    pred = base_model.predict(Test_X)
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = evaluation(test_target, pred)
    return Eval, pred
