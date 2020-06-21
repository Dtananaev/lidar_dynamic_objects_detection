#!/usr/bin/env python
__copyright__ = """
Copyright (c) 2020 Tananaev Denis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions: The above copyright notice and this permission
notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""


import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    Layer,
    UpSampling2D,
    BatchNormalization,
    LeakyReLU,
)
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2


class DarkNetConv2D(Layer):
    """
    The darknet conv layer yolo_v3
    """

    def __init__(
        self,
        filters,
        kernel,
        strides,
        padding,
        weight_decay,
        batch_norm=True,
        activation_funct=True,
        data_format="channels_last",
    ):
        super(DarkNetConv2D, self).__init__()
        self.batch_norm = batch_norm
        self.activation_funct = activation_funct
        self.conv = Conv2D(
            filters,
            kernel,
            strides=strides,
            activation=None,
            kernel_regularizer=l2(weight_decay),
            padding=padding,
            data_format=data_format,
        )
        self.bn = BatchNormalization()
        self.activation = LeakyReLU(alpha=0.1)

    def call(self, x, training=False):
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x, training=training)
        if self.activation_funct:
            x = self.activation(x)
        return x


class DarkNetBlock(Layer):
    """
    The darknet block layer
    """

    def __init__(
        self, filters, weight_decay, batch_norm=True, data_format="channels_last"
    ):
        super(DarkNetBlock, self).__init__()

        self.conv1 = DarkNetConv2D(
            filters // 2,
            1,
            strides=1,
            padding="same",
            weight_decay=weight_decay,
            batch_norm=batch_norm,
            data_format=data_format,
        )
        self.conv2 = DarkNetConv2D(
            filters,
            3,
            strides=1,
            padding="same",
            weight_decay=weight_decay,
            batch_norm=batch_norm,
            data_format=data_format,
        )

    def call(self, x, training=False):
        prev = x
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        return prev + x


class DarkNetDecoderBlock(Layer):
    """
    The yolo v3 decoder layer 
    """

    def __init__(self, filters, weight_decay, data_format="channels_last"):
        super(DarkNetDecoderBlock, self).__init__()

        self.conv1_1 = DarkNetConv2D(
            filters,
            (1, 1),
            strides=(1, 1),
            weight_decay=weight_decay,
            padding="same",
            data_format=data_format,
        )
        self.conv1_2 = DarkNetConv2D(
            filters * 2,
            (3, 3),
            strides=(1, 1),
            weight_decay=weight_decay,
            padding="same",
            data_format=data_format,
        )
        self.conv2_1 = DarkNetConv2D(
            filters,
            (1, 1),
            strides=(1, 1),
            weight_decay=weight_decay,
            padding="same",
            data_format=data_format,
        )
        self.conv2_2 = DarkNetConv2D(
            filters * 2,
            (3, 3),
            strides=(1, 1),
            weight_decay=weight_decay,
            padding="same",
            data_format=data_format,
        )
        self.conv3 = DarkNetConv2D(
            filters,
            (1, 1),
            strides=(1, 1),
            weight_decay=weight_decay,
            padding="same",
            data_format=data_format,
        )

    def call(self, x, training=False):
        x = self.conv1_1(x, training=training)
        x = self.conv1_2(x, training=training)
        x = self.conv2_1(x, training=training)
        x = self.conv2_2(x, training=training)
        x = self.conv3(x, training=training)
        return x


class DarkNetEncoder(Layer):
    """
    The darknet 53 encoder from yolo_v3
    See: https://arxiv.org/abs/1804.02767
    """

    def __init__(self, name, weight_decay, data_format="channels_last"):
        super(DarkNetEncoder, self).__init__(name=name)
        #  Input
        self.conv1 = DarkNetConv2D(
            32,
            (3, 3),
            strides=(1, 1),
            weight_decay=weight_decay,
            padding="same",
            data_format=data_format,
        )
        # Conv with stride 2
        self.conv2 = DarkNetConv2D(
            64,
            (3, 3),
            strides=(2, 2),
            weight_decay=weight_decay,
            padding="same",
            data_format=data_format,
        )
        # Residual block
        self.block_1 = DarkNetBlock(
            64, weight_decay=weight_decay, data_format=data_format
        )
        # Conv with stride 2
        self.conv3 = DarkNetConv2D(
            128,
            (3, 3),
            strides=(2, 2),
            weight_decay=weight_decay,
            padding="same",
            data_format=data_format,
        )
        # Residual blocks 2x
        self.block_2 = []
        for _ in range(2):
            self.block_2.append(
                DarkNetBlock(128, weight_decay=weight_decay, data_format=data_format)
            )
        # Conv with stride 2
        self.conv4 = DarkNetConv2D(
            256,
            (3, 3),
            strides=(2, 2),
            weight_decay=weight_decay,
            padding="same",
            data_format=data_format,
        )
        # Residual blocks 8x
        self.block_3 = []
        for _ in range(8):
            self.block_3.append(
                DarkNetBlock(256, weight_decay=weight_decay, data_format=data_format)
            )
        # Conv with stride 2
        self.conv5 = DarkNetConv2D(
            512,
            (3, 3),
            strides=(2, 2),
            weight_decay=weight_decay,
            padding="same",
            data_format=data_format,
        )
        # Residual blocks 8x
        self.block_4 = []
        for _ in range(8):
            self.block_4.append(
                DarkNetBlock(512, weight_decay=weight_decay, data_format=data_format)
            )
        # Conv with stride 2
        self.conv6 = DarkNetConv2D(
            1024,
            (3, 3),
            strides=(2, 2),
            weight_decay=weight_decay,
            padding="same",
            data_format=data_format,
        )
        # Residual blocks 4x
        self.block_5 = []
        for _ in range(4):
            self.block_5.append(
                DarkNetBlock(1024, weight_decay=weight_decay, data_format=data_format)
            )

    def call(self, x, training=False):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = x_b1 = self.block_1(x, training=training)
        x = self.conv3(x, training=training)
        for i in range(len(self.block_2)):
            x = x_b2 = self.block_2[i](x, training=training)
        x = self.conv4(x)
        for i in range(len(self.block_3)):
            x = x_b3 = self.block_3[i](x, training=training)
        x = self.conv5(x, training=training)
        for i in range(len(self.block_4)):
            x = x_b4 = self.block_4[i](x, training=training)
        x = self.conv6(x, training=training)
        for i in range(len(self.block_5)):
            x = x_b5 = self.block_5[i](x, training=training)

        return x_b5, x_b4, x_b3, x_b2, x_b1


class DarkNetDecoder(Layer):
    """
    The yolo v3 decoder 
    """

    def __init__(self, name, weight_decay, data_format="channels_last"):
        super(DarkNetDecoder, self).__init__(name=name)

        self.decoder_block_1 = DarkNetDecoderBlock(
            filters=512, weight_decay=weight_decay, data_format=data_format
        )

        self.conv1 = DarkNetConv2D(
            256,
            (1, 1),
            strides=(1, 1),
            weight_decay=weight_decay,
            padding="same",
            data_format=data_format,
        )
        self.up1 = UpSampling2D(size=(2, 2), data_format=data_format)
        self.decoder_block_2 = DarkNetDecoderBlock(
            filters=256, weight_decay=weight_decay, data_format=data_format
        )

        self.conv2 = DarkNetConv2D(
            128,
            (1, 1),
            strides=(1, 1),
            weight_decay=weight_decay,
            padding="same",
            data_format=data_format,
        )
        self.up2 = UpSampling2D(size=(2, 2), data_format=data_format)
        self.decoder_block_3 = DarkNetDecoderBlock(
            filters=128, weight_decay=weight_decay, data_format=data_format
        )
        self.up3 = UpSampling2D(size=(2, 2), data_format=data_format)
        self.decoder_block_4 = DarkNetDecoderBlock(
            filters=64, weight_decay=weight_decay, data_format=data_format
        )
        self.up4 = UpSampling2D(size=(2, 2), data_format=data_format)
        self.decoder_block_5 = DarkNetDecoderBlock(
            filters=64, weight_decay=weight_decay, data_format=data_format
        )

    def call(self, x_in, training=False):
        # First lvl
        x_b5, x_b4, x_b3, x_b2, x_b1 = x_in
        x = self.decoder_block_1(x_b5, training=training)
        # Second lvl
        x = self.conv1(x, training=training)
        x = self.up1(x)
        x = tf.concat([x, x_b4], axis=-1)
        x = self.decoder_block_2(x, training=training)
        # Third lvl
        x = self.conv2(x, training=training)
        x = self.up2(x)
        x = tf.concat([x, x_b3], axis=-1)
        x = self.decoder_block_3(x, training=training)
        x = self.up3(x)
        x = tf.concat([x, x_b2], axis=-1)
        x = self.decoder_block_4(x, training=training)
        x = self.up4(x)
        x = tf.concat([x, x_b1], axis=-1)
        x = self.decoder_block_5(x, training=training)

        return x


class YoloV3_Lidar(Model):
    def __init__(self, weight_decay, num_classes=26, data_format="channels_last"):
        super(YoloV3_Lidar, self).__init__(name="YoloV3_Lidar")
        self.encoder = DarkNetEncoder(
            name="DarkNetEncoder", weight_decay=weight_decay, data_format=data_format
        )
        self.decoder = DarkNetDecoder(
            name="DarkNetDecoder", weight_decay=weight_decay, data_format=data_format
        )
        self.final_layer = Conv2D(
            8 + num_classes,
            (1, 1),
            activation=None,
            padding="same",
            data_format=data_format,
        )

    def call(self, x, training):
        x = self.encoder(x, training=training)
        x = self.decoder(x, training=training)
        x = self.final_layer(x)
        return x
