# Importing the libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from Evaluation import evaluation


def Model_Bi_GRU(Train_Data, Train_Target, Test_Data, Test_Target, epoch):
    X_train = np.reshape(Train_Data, (Train_Data.shape[0], Train_Data.shape[1], 1))
    TestX = np.reshape(Test_Data, (Test_Data.shape[0], Test_Data.shape[1], 1))
    # The GRU architecture
    regressorGRU = Sequential()
    # First GRU layer with Dropout regularisation
    regressorGRU.add(Bidirectional(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='tanh')))
    regressorGRU.add(Dropout(0.2))
    # Second GRU layer
    regressorGRU.add(Bidirectional(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='tanh')))
    regressorGRU.add(Dropout(0.2))
    # Third GRU layer
    regressorGRU.add(Bidirectional(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='tanh')))
    regressorGRU.add(Dropout(0.2))
    # Fourth GRU layer
    regressorGRU.add(Bidirectional(GRU(units=50, activation='tanh')))
    regressorGRU.add(Dropout(0.2))
    # The output layer
    regressorGRU.add(Dense(units=Train_Target.shape[1]))
    regressorGRU.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    TrainX = np.zeros((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    for k in range(X_train.shape[0]):
        for j in range(X_train.shape[2]):
            val = X_train[k, :, j][0]
            TrainX[k, :, j] = np.asarray(val).astype(np.float32)
    TestX = np.zeros((TestX.shape[0], TestX.shape[1], TestX.shape[2]))
    for k in range(TestX.shape[0]):
        for j in range(TestX.shape[2]):
            val = TestX[k, :, j][0]
            TestX[k, :, j] = np.asarray(val).astype(np.float32)
    # Fitting to the training set
    regressorGRU.fit(TrainX, Train_Target, epochs=epoch, steps_per_epoch=5)
    pred = regressorGRU.predict(TestX)
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = evaluation(Test_Target, pred)

    return Eval, pred