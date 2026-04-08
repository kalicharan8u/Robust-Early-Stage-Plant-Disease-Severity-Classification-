import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers
from keras import backend as K


def Model_Feat_ViT(Image, Target):
    IMG_SIZE = 256
    images = np.zeros((Image.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Image.shape[0]):
        temp = np.resize(Image[i], (IMG_SIZE * IMG_SIZE, 3))
        images[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    inputs = tf.keras.Input(shape=(images.shape[1:]))
    # Patch embedding
    x = layers.Conv2D(64, kernel_size=16, strides=16, activation="relu")(inputs)
    x = layers.Flatten()(x)
    # Positional embeddings
    x = layers.Embedding(input_dim=16 * 16, output_dim=64)(x)
    # Transformer encoder
    transformer_block = layers.MultiHeadAttention(num_heads=8, key_dim=64, dropout=0.1)
    x = transformer_block(x, x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(Target.shape[1], activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(Image.shape[0], activation="softmax")(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(images, Target, epochs=50)
    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions
    layer_no = 5
    layer_out = np.asarray(functors[layer_no]([Image])).squeeze()
    return layer_out

