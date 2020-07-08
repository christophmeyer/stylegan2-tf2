import tensorflow as tf
import numpy as np

from model.layers import FullConvLayer, Embedding, NormalizePixels, DenseMod, LeakyReLU, Broadcast
from model.layers import ToRGB, UpSampling2D
from model.utils import num_feature_maps_builder


class GeneratorBlock(tf.keras.layers.Layer):
    """Block with two conv layers of the generator network
    """

    def __init__(self, filters, alpha):
        super().__init__()
        self.full_conv_1_up = FullConvLayer(filters=filters,
                                            kernel_size=3,
                                            upsample=True,
                                            modulate=True,
                                            demodulate=True,
                                            add_bias=True,
                                            add_noise=True,
                                            apply_act=True,
                                            alpha=alpha)

        self.full_conv_2 = FullConvLayer(filters=filters,
                                         kernel_size=3,
                                         modulate=True,
                                         demodulate=True,
                                         add_bias=True,
                                         add_noise=True,
                                         apply_act=True,
                                         alpha=alpha)

    def call(self, x, dlatent_1, dlatent_2):
        x = self.full_conv_1_up(x, dlatent_1)
        x = self.full_conv_2(x, dlatent_2)
        return x


class GeneratorMapping(tf.keras.Model):
    """Mapping network of the generator
    """

    def __init__(self, cfg):
        super().__init__()
        self.label_conditioning = cfg.label_conditioning
        if cfg.label_conditioning:
            self.label_embedding = Embedding(input_dim=cfg.labels_size,
                                             output_dim=cfg.latent_size)
        # Calculate number of layers from resolution for latent broadcast
        num_layers = int(np.log2(cfg.resolution)) * 2 - 2
        # Build model sequentially
        model = tf.keras.Sequential()
        if cfg.normalize_latents:
            model.add(NormalizePixels())
        for layer_id in range(cfg.num_dense_layers):
            units = cfg.dlatent_size if layer_id == cfg.num_dense_layers - 1 else cfg.hidden_size
            model.add(DenseMod(units=units,
                               lr_multiplier=cfg.lr_multiplier))
            model.add(LeakyReLU(alpha=cfg.alpha, gain=np.sqrt(2)))
        model.add(Broadcast(dlatent_broadcast=num_layers))
        self.model = model

    def call(self, inputs, labels=None):
        if self.label_conditioning:
            y = self.label_embedding(labels)
            inputs = tf.concat([inputs, y], axis=1)
        return self.model(inputs)


class GeneratorSynthesis(tf.keras.Model):
    """Synthesis network for config E with skip architecture
    """

    def __init__(self, cfg):
        super().__init__()
        self.num_channels = cfg.num_channels
        self.resolution_log2 = int(np.log2(cfg.resolution))
        self.num_layers = self.resolution_log2 * 2 - 2
        self.num_feature_maps = num_feature_maps_builder(cfg.feature_maps_base,
                                                         cfg.feature_maps_decay,
                                                         cfg.feature_maps_min,
                                                         cfg.feature_maps_max)

        self.constant_input = self.add_weight(name='constant_input', shape=[1, 4, 4, self.num_feature_maps(1)],
                                              initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))

        self.first_full_conv = FullConvLayer(filters=self.num_feature_maps(1),
                                             kernel_size=3,
                                             modulate=True,
                                             demodulate=True,
                                             add_noise=True,
                                             add_bias=True,
                                             apply_act=True,
                                             alpha=cfg.alpha)

        self.first_to_rgb = ToRGB(filters=self.num_channels, alpha=cfg.alpha)

        self.conv_blocks = [GeneratorBlock(filters=self.num_feature_maps(n_layer), alpha=cfg.alpha)
                            for n_layer in range(3, self.resolution_log2 + 1)]

        self.to_rgb_layers = [ToRGB(filters=self.num_channels, alpha=cfg.alpha)
                              for n_layer in range(3, self.resolution_log2 + 1)]
        self.upsample2d = UpSampling2D(size=(2, 2),
                                       data_format='channels_last',
                                       interpolation='bilinear')

    def call(self, inputs):
        dlatents = inputs
        assert dlatents.shape[1] == self.num_layers

        # First two conv layers
        # Copy constant input for each example in minibatch
        x = tf.tile(self.constant_input, [tf.shape(dlatents)[0], 1, 1, 1])
        x = self.first_full_conv(x, dlatents[:, 0, :])
        y = self.first_to_rgb(x, None, dlatents[:, 1, :])
        # Main layers

        for n_layer in range(3, self.resolution_log2 + 1):
            x = self.conv_blocks[n_layer-3](x,
                                            dlatents[:, n_layer * 2 - 5, :],
                                            dlatents[:, n_layer * 2 - 4, :])
            y = self.upsample2d(y)
            y = self.to_rgb_layers[n_layer-3](x, y, dlatents[:, n_layer*2-3, :])

        return y


class Generator(tf.keras.Model):
    """Generator network for config E with skip architecture
    """
    # TODO:
    # - style mixing regularization

    def __init__(self, cfg):
        super().__init__()
        self.mapping_network = GeneratorMapping(cfg)
        self.synthesis_network = GeneratorSynthesis(cfg)
        self.dlatent_moving_avg = tf.Variable(initial_value=tf.zeros([cfg.dlatent_size]),
                                              trainable=False)
        self.dlatent_avg_beta = cfg.dlatent_avg_beta
        if cfg.truncation_psi is not None:
            self.truncation_psi = cfg.truncation_psi
        if cfg.truncation_cutoff is not None:
            self.truncation_cutoff = cfg.truncation_cutoff

    def call(self, inputs, labels=None, training=False):
        # Map latents to dlatents
        dlatents = self.mapping_network(inputs, labels)

        # Update moving average
        if training:
            batch_avg = tf.reduce_mean(dlatents[:, 0], axis=0)
            self.dlatent_moving_avg = batch_avg + (self.dlatent_moving_avg-batch_avg)*self.dlatent_avg_beta

        # Apply truncation trick for all layers below truncation_cutoff
        if (not training) and (self.truncation_psi is not None):
            layer_idx = np.arange(dlatents.shape[1])[np.newaxis, :, np.newaxis]
            layer_psi = tf.ones(layer_idx.shape)
            if self.truncation_cutoff is None:
                layer_psi *= self.truncation_psi
            else:
                layer_psi = tf.where(layer_idx < self.truncation_cutoff, layer_psi * self.truncation_psi, layer_psi)
            dlatents = self.dlatent_moving_avg + (dlatents - self.dlatent_moving_avg)*layer_psi

        images = self.synthesis_network(dlatents)

        return images
