import tensorflow as tf
import numpy as np

from model.layers import FullConvLayer, FromRGB, DenseMod, Minibatch_Std_Layer, LeakyReLU
from model.utils import num_feature_maps_builder


class DiscriminatorBlock(tf.keras.layers.Layer):
    """Block of the discriminator network for config E with resnet architecture
    """

    def __init__(self, filters, filters_inter, alpha):
        super().__init__()
        self.full_conv_1 = FullConvLayer(filters=filters_inter,
                                         kernel_size=3,
                                         add_bias=True,
                                         apply_act=True,
                                         alpha=alpha)

        self.full_conv_2_down = FullConvLayer(filters=filters,
                                              kernel_size=3,
                                              downsample=True,
                                              add_bias=True,
                                              apply_act=True,
                                              alpha=alpha)

        self.full_conv_res_down = FullConvLayer(filters=filters,
                                                kernel_size=1,
                                                downsample=True,
                                                alpha=alpha)

    def call(self, x):
        t = x
        x = self.full_conv_1(x)
        x = self.full_conv_2_down(x)
        t = self.full_conv_res_down(t)
        x = (x+t) / np.sqrt(2)

        return x


class Discriminator(tf.keras.Model):
    """Discriminator network for config E with resnet architecture
    """

    def __init__(self, cfg):
        super().__init__()

        self.num_channels = cfg.num_channels
        self.label_conditioning = cfg.label_conditioning
        self.resolution_log2 = int(np.log2(cfg.resolution))
        self.num_feature_maps = num_feature_maps_builder(cfg.feature_maps_base,
                                                         cfg.feature_maps_decay,
                                                         cfg.feature_maps_min,
                                                         cfg.feature_maps_max)

        self.from_rgb = FromRGB(filters=self.num_feature_maps(self.resolution_log2-1), alpha=cfg.alpha)

        self.blocks = [DiscriminatorBlock(filters=self.num_feature_maps(n_layer),
                                          filters_inter=self.num_feature_maps(n_layer - 1),
                                          alpha=cfg.alpha)
                       for n_layer in range(self.resolution_log2, 2, -1)]
        self.minibatch_std_layer = Minibatch_Std_Layer(group_size=cfg.minibatch_std_group_size,
                                                       n_new_features=cfg.minibatch_std_n_features)

        self.full_conv_final = FullConvLayer(filters=self.num_feature_maps(1),
                                             kernel_size=3,
                                             add_bias=True,
                                             apply_act=True,
                                             alpha=cfg.alpha)
        self.dense_final_hidden = DenseMod(units=self.num_feature_maps(0))
        self.activation = LeakyReLU(alpha=cfg.alpha, gain=np.sqrt(2))
        self.dense_output = DenseMod(units=max(1, cfg.labels_size))

    def call(self, inputs, labels=None):

        # Main layers
        x = None
        y = inputs
        for n_layer in range(self.resolution_log2, 2, -1):
            if n_layer == self.resolution_log2:
                x = self.from_rgb(x, y)
            x = self.blocks[self.resolution_log2-n_layer-1](x)

        # Final layers
        x = self.minibatch_std_layer(x)
        x = self.full_conv_final(x)
        x = self.dense_final_hidden(x)
        x = self.activation(x)

        # Output layer
        x = self.dense_output(x)
        if self.label_conditioning:
            x = tf.reduce_sum(x * labels, axis=1, keepdims=True)

        return x
