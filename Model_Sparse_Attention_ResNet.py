import numpy as np
from keras import layers, models
from keras import backend as K
import warnings

warnings.filterwarnings('ignore')


# Define Sparse Attention as a function
def sparse_attention(inputs, num_heads=1):
    queries = layers.Dense(inputs.shape[-1])(inputs)
    keys = layers.Dense(inputs.shape[-1])(inputs)
    values = layers.Dense(inputs.shape[-1])(inputs)

    attention_scores = K.batch_dot(queries, keys, axes=[2, 2])
    mask = (K.ones_like(attention_scores))
    attention_scores = attention_scores * mask
    attention_weights = K.softmax(attention_scores, axis=-1)
    output = K.batch_dot(attention_weights, values)

    return output


def Model_Sparse_attention_ResNet(f1, f2, f3, Target):
    Feat1 = np.reshape(f1, (f1.shape[0], 1, 1, f1.shape[1]))
    Feat2 = np.reshape(f2, (f2.shape[0], 1, 1, f2.shape[1]))
    Feat3 = np.reshape(f3, (f3.shape[0], 1, 1, f3.shape[1]))

    input_layer1 = layers.Input(shape=(1, 1, Feat1.shape[3]))
    input_layer2 = layers.Input(shape=(1, 1, Feat2.shape[3]))
    input_layer3 = layers.Input(shape=(1, 1, Feat3.shape[3]))
    # Multi-scale convolution blocks
    scale_1 = layers.Conv2D(8, (3, 3), padding='same', activation='relu')(input_layer1)
    scale_2 = layers.Conv2D(9, (5, 5), padding='same', activation='relu')(input_layer2)
    scale_3 = layers.Conv2D(32, (7, 7), padding='same', activation='relu')(input_layer3)
    # Combine the multi-scale features
    combined = layers.concatenate([scale_1, scale_2, scale_3], axis=-1)  # Concatenate along the channels axis
    # Apply Sparse Attention as a function
    sparse_attention_output = sparse_attention(combined, num_heads=4)
    # Flatten the output of the attention mechanism
    x = layers.Flatten()(sparse_attention_output)
    # Fully connected layers for classification
    x = layers.Dense(Target.shape[1], activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    # Final classification layer
    outputs = layers.Dense(Target.shape[1], activation='softmax')(x)
    # Model with input and output layers
    model = models.Model(inputs=[input_layer1, input_layer2, input_layer3], outputs=outputs)
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Print model summary
    model.summary()
    # Fit the model
    model.fit([Feat1, Feat2, Feat3], Target, epochs=1, batch_size=32)
    inp = model.input
    outputs = [layer.output for layer in model.layers]
    functors = [K.function([inp], [out]) for out in outputs]
    layer = 8
    test = Feat1[:][np.newaxis, ...]
    layer_out = np.asarray(functors[layer]([test])).squeeze()

    return layer_out
