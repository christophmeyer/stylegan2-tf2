
import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from preprocessing.utils import plot_batch
from model.discriminator import Discriminator
from model.generator import Generator
from model.utils import ModelConfig

# pylint: disable=no-member


def generate_fakes(config_path, num_fake_batches, checkpoint_dir, generated_images_dir):
    cfg = ModelConfig(config_path)

    discriminator = Discriminator(cfg)
    generator = Generator(cfg)

    # Initialize models
    generator(tf.ones([cfg.batch_size, cfg.latent_size]), tf.ones([cfg.batch_size, cfg.labels_size]))
    discriminator(tf.ones([cfg.batch_size, cfg.resolution, cfg.resolution,
                           cfg.num_channels]), tf.ones([cfg.batch_size, cfg.labels_size]))

    # Initialize checkpoint and manager
    checkpoint = tf.train.Checkpoint(discriminator=discriminator, generator=generator)
    manager = tf.train.CheckpointManager(
        checkpoint, checkpoint_dir, max_to_keep=10)

    if cfg.label_conditioning:
        fake_labels = np.zeros([cfg.batch_size, cfg.labels_size], dtype=np.float32)
        fake_labels[np.arange(cfg.batch_size), np.random.randint(cfg.labels_size, size=cfg.batch_size)] = 1.0
    random_input = tf.random.normal([cfg.batch_size, cfg.latent_size])

    for checkpoint_path in manager.checkpoints:
        print('Restoring checkpoint from {}'.format(checkpoint_path))
        checkpoint.restore(checkpoint_path).assert_consumed()
        fake_image_batches = []
        fake_labels_batches = []
        for _ in range(num_fake_batches):
            if cfg.label_conditioning:
                fake_labels_batch = np.zeros([cfg.batch_size, cfg.labels_size], dtype=np.float32)
                fake_labels_batch[np.arange(cfg.batch_size), np.random.randint(
                    cfg.labels_size, size=cfg.batch_size)] = 1.0
            else:
                fake_labels_batch = None
            random_input = tf.random.normal([cfg.batch_size, cfg.latent_size])

            fake_images_batch = generator(random_input, fake_labels_batch)

            fake_image_batches.append(fake_images_batch)
            fake_labels_batches.append(fake_labels_batch)

        fake_images = np.concatenate(fake_image_batches, axis=0)
        if cfg.label_conditioning:
            fake_labels = np.concatenate(fake_labels_batches, axis=0)
        else:
            fake_labels = None

        if not os.path.exists(generated_images_dir):
            os.makedirs(generated_images_dir)

        plot_batch(fake_images, fake_labels, generated_images_dir + '/checkpoint-' +
                   checkpoint_path.split('-')[-1], cfg.label_conditioning)
