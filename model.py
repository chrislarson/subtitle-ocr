import keras.backend as K
from keras import layers
from keras.layers import (
    Dense,
    LSTM,
    BatchNormalization,
    Conv2D,
    MaxPool2D,
    Lambda,
    Bidirectional,
)
from keras.models import Model


def crnn(input_dim, output_dim, activation="relu", dropout=0.2):
    inputs = layers.Input(shape=input_dim, name="input")

    # Layer 1
    conv_1 = Conv2D(64, (3, 3), activation=activation, padding="same")(inputs)
    pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

    # Layer 2
    conv_2 = Conv2D(128, (3, 3), activation=activation, padding="same")(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

    # Layer 3
    conv_3 = Conv2D(256, (3, 3), activation=activation, padding="same")(pool_2)

    # Layer 4
    conv_4 = Conv2D(256, (3, 3), activation=activation, padding="same")(conv_3)
    pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

    # Layer 5
    conv_5 = Conv2D(512, (3, 3), activation=activation, padding="same")(pool_4)
    batch_norm_5 = BatchNormalization()(conv_5)

    # Layer 6
    conv_6 = Conv2D(512, (3, 3), activation=activation, padding="same")(batch_norm_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

    # Layer 7
    conv_7 = Conv2D(512, (2, 2), activation=activation)(pool_6)
    squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

    blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=dropout))(squeezed)
    blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout=dropout))(blstm_1)

    outputs = Dense(output_dim + 1, activation="softmax")(blstm_2)
    model = Model(inputs=inputs, outputs=outputs)

    return model
