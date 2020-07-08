import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, UpSampling2D, Layer, AveragePooling2D


class LeakyReLU(tf.keras.layers.Layer):
    def __init__(self, alpha, gain):
        super(LeakyReLU, self).__init__()
        self.alpha = alpha
        self.gain = gain
        self.activation = tf.keras.layers.LeakyReLU(alpha=self.alpha)

    def call(self, inputs):
        x = self.activation(inputs)
        x *= self.gain
        return x


def runtime_coef(shape, gain, lr_multiplier):
    fan_in = np.prod(shape[:-1])
    he_std = gain / np.sqrt(float(fan_in))
    return he_std * lr_multiplier


class DenseMod(Layer):
    def __init__(self,
                 units,
                 gain=1.0,
                 lr_multiplier=1.0,
                 bias_initializer=tf.zeros_initializer()):

        super().__init__()
        self.units = units
        self.kernel_initializer = tf.random_normal_initializer(mean=0.0,
                                                               stddev=1 / lr_multiplier)
        self.bias_initializer = bias_initializer
        self.gain = gain
        self.lr_multiplier = lr_multiplier

    def build(self, input_shape):

        self.kernel = self.add_weight(name='kernel', shape=[np.prod(input_shape[1:]), self.units],
                                      initializer=self.kernel_initializer)
        self.bias = self.add_weight(name='bias', shape=[self.units],
                                    initializer=self.bias_initializer)

    def call(self, inputs):
        s = inputs.shape
        if len(s) > 2:
            x = tf.reshape(inputs, [-1, np.prod(s[1:])])
        else:
            x = inputs
        bias = self.bias * self.lr_multiplier
        kernel = self.kernel * runtime_coef(self.kernel.shape, self.gain, self.lr_multiplier)
        out = tf.matmul(x, kernel)
        out += bias
        return out


class NormalizePixels(Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        inputs *= tf.math.rsqrt(tf.reduce_mean(tf.square(inputs), axis=1, keepdims=True) + 1e-8)
        return inputs


class Broadcast(Layer):
    def __init__(self, dlatent_broadcast):
        super().__init__()
        self.dlatent_broadcast = dlatent_broadcast

    def call(self, inputs):
        return tf.tile(inputs[:, np.newaxis], [1, self.dlatent_broadcast, 1])


class Embedding(Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.kernel = self.add_weight(name='kernel', shape=[input_dim, output_dim],
                                      initializer=tf.random_normal_initializer())

    def call(self, x):
        x = tf.matmul(x, self.kernel)
        return x


def grouped_conv2d(inputs, filters, data_format):
    """
    Grouped convolution. Number of input channels must be divisible by the number of groups.
    """
    # Use naive loop-based implementation on CPU
    if inputs.shape[3] % filters.shape[2] != 0:
        raise ValueError('Number of input channels must be divisible by number of groups.')
    else:
        num_groups = inputs.shape[3] // filters.shape[2]

    if len(tf.config.list_physical_devices('GPU')) == 0:
        inputs = tf.split(inputs, num_groups, axis=1 if data_format == 'NCHW' else 3)
        filters = tf.split(filters, num_groups, axis=3)
        output = tf.concat(
            [tf.nn.conv2d(i, f,
                          strides=[1, 1, 1, 1],
                          padding='SAME',
                          data_format=data_format) for i, f in zip(inputs, filters)], axis=1 if data_format == 'NCHW' else 3)
    else:
        # On GPU tf.nn.conv2d supports grouped convolutions since
        # https://github.com/tensorflow/tensorflow/pull/25818
        output = tf.nn.conv2d(inputs, filters, strides=[1, 1, 1, 1], padding='SAME', data_format=data_format)
    return output


class FullConvLayer(Layer):
    """
    Convolutional layer with the following features: modulation/demodulation, upsampling/downsampling,
    noise, bias, leaky relu activation.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 lr_multiplier=1,
                 gain=1,
                 modulate=False,
                 demodulate=False,
                 upsample=False,
                 downsample=False,
                 add_noise=False,
                 add_bias=False,
                 apply_act=False,
                 alpha=0.2):
        super().__init__()
        assert not (upsample and downsample)

        self.add_noise = add_noise
        self.add_bias = add_bias
        self.apply_act = apply_act
        self.upsample = upsample
        self.downsample = downsample

        self.conv2d_mod = Conv2DModulated(filters=filters,
                                          kernel_size=kernel_size,
                                          lr_multiplier=lr_multiplier,
                                          gain=gain,
                                          modulate=modulate,
                                          demodulate=demodulate)
        if self.add_noise:
            self.noise_strength = self.add_weight(name='noise_strength', shape=[], initializer='zeros')

        if self.add_bias:
            self.bias = self.add_weight(name='bias', shape=[filters], initializer='zeros')

        # Init upsample layer
        if self.upsample:
            self.upsample2d = UpSampling2D(size=(2, 2),
                                           data_format='channels_last',
                                           interpolation='bilinear')

        # Init downsampling layer
        if self.downsample:
            self.downsample2d = AveragePooling2D(pool_size=(2, 2),
                                                 data_format='channels_last')

        if self.apply_act:
            self.activation = LeakyReLU(alpha=alpha, gain=np.sqrt(2))

    def call(self, x, dlatents=None):
        # Upsample
        if self.upsample:
            x = self.upsample2d(x)

        # Downsample
        if self.downsample:
            x = self.downsample2d(x)

        # Apply conv2d with modulation
        x = self.conv2d_mod(x, dlatents)

        # same noise image for all out-channels
        if self.add_noise:
            noise = tf.random.normal([x.shape[0], x.shape[1], x.shape[2], 1])
            x += noise * tf.cast(self.noise_strength, tf.float32)

        # Add bias
        if self.add_bias:
            x += tf.reshape(self.bias, [1, 1, 1, -1])

        # Apply activation
        if self.apply_act:
            x = self.activation(x)
        return x


class FromRGB(Layer):

    def __init__(self, filters, alpha):
        super().__init__()
        self.conv2d = FullConvLayer(filters=filters,
                                    kernel_size=1,
                                    add_bias=True,
                                    apply_act=True,
                                    alpha=alpha)

    def call(self, x, y):
        t = self.conv2d(y)
        return t if x is None else x + t


class ToRGB(Layer):

    def __init__(self, filters, alpha):
        super().__init__()
        self.conv2d_mod = FullConvLayer(filters=filters,
                                        kernel_size=1,
                                        modulate=True,
                                        add_bias=True,
                                        alpha=alpha)

    def call(self, x, y, dlatents):
        x = self.conv2d_mod(x, dlatents)
        y = x if y is None else y + x
        return y


class Minibatch_Std_Layer(Layer):
    def __init__(self, group_size, n_new_features):
        super().__init__()
        self.group_size = group_size
        self.n_new_features = n_new_features

    def call(self, x):
        x = tf.transpose(x, [0, 3, 1, 2])
        s = x.shape  # Tranposed input shape: [BCHW]
        y = tf.reshape(x, [self.group_size, -1, self.n_new_features, s[1]//self.n_new_features,
                           s[2], s[3]])  # [GMncHW]
        y -= tf.reduce_mean(y, axis=0, keepdims=True)  # Subtract mean over group
        y = tf.sqrt(tf.reduce_mean(tf.square(y), axis=0) + 1e-8)  # [MncHW] Calc stddev over group
        y = tf.reduce_mean(y, axis=[2, 3, 4], keepdims=True)  # [Mn111] take mean over feature maps and pixels
        y = tf.reduce_mean(y, axis=[2])  # [Mn11]
        y = tf.tile(y, [self.group_size, 1, s[2], s[3]])  # [BCHW] copy over group and pixel indices
        return tf.transpose(tf.concat([x, y], axis=1), [0, 2, 3, 1])


class Conv2DModulated(Layer):
    """Convolutional layer with modulation/demodulation. 
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 lr_multiplier=1,
                 gain=1,
                 modulate=True,
                 demodulate=True):
        assert not (demodulate and not modulate)

        super(Conv2DModulated, self).__init__()
        self.lr_multiplier = lr_multiplier
        self.gain = gain
        self.kernel_size = kernel_size
        self.filters = filters
        self.demodulate = demodulate
        self.modulate = modulate

    def build(self, input_shape):
        channels = input_shape[3]
        self.kernel = self.add_weight(name='kernel', shape=[self.kernel_size, self.kernel_size, channels, self.filters],
                                      initializer=tf.random_normal_initializer(mean=0.0,
                                                                               stddev=1.0 / self.lr_multiplier))
        self.dense_layer = DenseMod(units=channels,
                                    bias_initializer=tf.ones_initializer())

    def call(self, inputs, dlatents=None):
        assert not (self.modulate and (dlatents is None))
        w = self.kernel * runtime_coef(self.kernel.shape, self.gain, self.lr_multiplier)  # [kkIO]

        x = inputs
        assert x.shape[3] == w.shape[2]

        if self.modulate:
            style = self.dense_layer(dlatents)  # [BI]

            # Introduce new batch dimension for kernel
            w = w[np.newaxis]  # [Bkkcf]

            # Incorporate per example modulation in convolution kernel
            w = w * style[:, np.newaxis, np.newaxis, :, np.newaxis]  # [BkkIO]

            # Incorporate demodulation in convolution kernel as well
            if self.demodulate:
                sig = tf.math.rsqrt(tf.reduce_sum(tf.square(w), axis=[1, 2, 3])+1e-8)  # [BO]
                w = w * sig[:, np.newaxis, np.newaxis, np.newaxis, :]  # [BkkIO]

            # Reshape examples into inputs channels
            x = x[np.newaxis]  # [1BHWI]
            x = tf.transpose(x, [0, 2, 3, 1, 4])  # [1HWBI]
            x = tf.reshape(x, [1, x.shape[1], x.shape[2], -1])  # [1HW(B*I)]

            w = tf.transpose(w, [1, 2, 3, 0, 4])  # [kkIBO]
            w = tf.reshape(w, [w.shape[0], w.shape[1], w.shape[2], -1])  # [kkI(B*O)]

        # Apply convolution
        x = grouped_conv2d(x, w, data_format='NHWC')

        if self.modulate:
            # Reshape output back into [BHWO]
            x = tf.reshape(x, [x.shape[1], x.shape[2], -1, self.kernel.shape[3]])  # [HWBO]
            x = tf.transpose(x, [2, 0, 1, 3])  # [BHWO]

        return x
