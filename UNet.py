import os

import imageio
from PIL import Image

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split

def encoder_block(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(inputs)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(conv)

    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)
    if max_pooling:
        next_layer = MaxPooling2D(pool_size=(2,2))(conv)
    else:
        next_layer = conv
    skip_connection = conv
    return next_layer, skip_connection

def decoder_block(prev_layer_input, skip_layer_input, n_filters=32):
    upsample = Conv2DTranspose(n_filters, (3,3), strides=(2,2), padding='same')(prev_layer_input)
    concat = concatenate([upsample, skip_layer_input], axis=3)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concat)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    return conv

def UNet(input_size=(128, 128, 3), n_filters=32, n_classes=255):
    inputs = Input(input_size)
    encoder0 = encoder_block(inputs, n_filters, max_pooling=True, dropout_prob=0)
    encoder1 = encoder_block(encoder0[0], n_filters*2, max_pooling=True, dropout_prob=0)
    encoder2 = encoder_block(encoder1[0], n_filters*4, max_pooling=True, dropout_prob=0)
    encoder3 = encoder_block(encoder2[0], n_filters*8, max_pooling=True, dropout_prob=0.3)
    encoder4 = encoder_block(encoder3[0], n_filters*16, max_pooling=False, dropout_prob=0.3)

    decoder_block5 = decoder_block(encoder4[0], encoder3[1], n_filters*8)
    decoder_block6 = decoder_block(decoder_block5, encoder2[1], n_filters*4)
    decoder_block7 = decoder_block(decoder_block6, encoder1[1], n_filters*2)
    decoder_block8 = decoder_block(decoder_block7, encoder0[1], n_filters)

    # output = Conv2D(n_classes, (1, 1), activation='softmax', padding='same')(decoder_block8)
    output = Conv2D(1, (1, 1), activation='linear', padding='same')(decoder_block8)
    # conv9 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(decoder_block8)
    # conv10 = Conv2D(n_classes, 1, padding='same')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    
    return model