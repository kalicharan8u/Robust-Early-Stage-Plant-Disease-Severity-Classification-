import numpy as np
from Evaluation import evaluation
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense
from keras.models import Model


def residual_block(input_tensor, filters, dilation_rate=1):
    """
    Residual block with dilated convolutions.
    """
    x = Conv2D(filters, (3, 3), padding='same', dilation_rate=dilation_rate)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3, 3), padding='same', dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x


def dilated_residual_attention_network(input_shape, num_classes=1):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = residual_block(x, 32, dilation_rate=1)
    x = residual_block(x, 32, dilation_rate=2)
    x = residual_block(x, 32, dilation_rate=4)
    x = Conv2D(64, (3, 3), padding='same', dilation_rate=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', dilation_rate=4)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    if num_classes == 1:
        outputs = Dense(1, activation='sigmoid')(x)  # Binary classification
    else:
        outputs = Dense(num_classes, activation='softmax')(x)  # Multi-class classification
    model = Model(inputs, outputs)
    return model


def Model_RAN(Train_Data, Train_Target, Test_data, Test_Target, epochs):
    X_train = np.reshape(Train_Data, (Train_Data.shape[0], Train_Data.shape[1], 1, 1))
    TestX = np.reshape(Test_data, (Test_data.shape[0], Test_data.shape[1], 1, 1))
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = dilated_residual_attention_network(input_shape, num_classes=Train_Target.shape[1])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(X_train, Train_Target, epoch=epochs)
    pred = model.predict(TestX)
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = evaluation(Test_Target, pred)

    return Eval, pred
