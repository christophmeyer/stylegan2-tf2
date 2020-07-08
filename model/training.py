import time

import tensorflow as tf
import numpy as np

from model.generator import Generator
from model.discriminator import Discriminator
from model.utils import ModelConfig
from preprocessing.dataset import read_dataset

# pylint: disable=no-member


def gen_logistic_non_sat_loss(fake_scores):
    return tf.reduce_mean(tf.nn.softplus(-fake_scores))


def disc_logistic_loss(fake_scores, real_scores):
    return tf.reduce_mean(tf.nn.softplus(fake_scores) + tf.nn.softplus(-real_scores))


def disc_logistic_loss_r1(fake_scores, real_scores, cfg, disc_real_score_gradients=None):
    loss = disc_logistic_loss(fake_scores, real_scores)
    if disc_real_score_gradients is not None:
        per_example_gradient_penalty = cfg.gamma * 0.5 * \
            tf.reduce_sum(tf.square(disc_real_score_gradients), axis=[1, 2, 3])
        gradient_penalty = tf.reduce_mean(per_example_gradient_penalty)
        loss += gradient_penalty
    return loss


def train_step(real_images,
               real_labels,
               generator,
               discriminator,
               gen_optimizer,
               disc_optimizer,
               cfg,
               labels=None,
               disc_regularization=True):

    latents = tf.random.normal([real_images.shape[0], cfg.dlatent_size])
    if cfg.labels_size > 0:
        fake_labels = np.zeros(real_labels.shape, dtype=np.float32)
        fake_labels[np.arange(cfg.batch_size), np.random.randint(10, size=cfg.batch_size)] = 1.0
    else:
        fake_labels = []

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        fake_images = generator(latents, labels=fake_labels, training=True)
        fake_scores = discriminator(fake_images, fake_labels)

        if disc_regularization:
            with tf.GradientTape() as inner_tape:
                inner_tape.watch(real_images)
                real_scores = discriminator(real_images, labels=real_labels)
                disc_real_score_gradients = inner_tape.gradient(tf.reduce_sum(real_scores), real_images)
                disc_loss = disc_logistic_loss_r1(fake_scores, real_scores, cfg, disc_real_score_gradients)
        else:
            real_scores = discriminator(real_images, labels=real_labels)
            disc_loss = disc_logistic_loss_r1(fake_scores, real_scores, cfg)

        gen_loss = gen_logistic_non_sat_loss(fake_scores)

    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    return gen_loss, disc_loss


def save_checkpoint(ckpt_manager, num_images):
    checkpoint_path = ckpt_manager.save(checkpoint_number=num_images)
    print('Saved checkpoint to {}'.format(checkpoint_path))


def train_model(config_path, data_path):

    cfg = ModelConfig(config_path)

    discriminator = Discriminator(cfg)
    generator = Generator(cfg)

    gen_optimizer = tf.keras.optimizers.Adam(
        learning_rate=cfg.generator_base_learning_rate,
        beta_1=cfg.generator_beta_1,
        beta_2=cfg.generator_beta_2,
        epsilon=1e-8)

    disc_optimizer = tf.keras.optimizers.Adam(
        learning_rate=cfg.discriminator_base_learning_rate,
        beta_1=cfg.discriminator_beta_1,
        beta_2=cfg.discriminator_beta_2,
        epsilon=1e-8)

    dataset = read_dataset(data_path, cfg)

    # Initialize checkpoint and checkpoint manager
    ckpt_models = tf.train.Checkpoint(generator=generator,
                                      discriminator=discriminator)

    ckpt_optimizers = tf.train.Checkpoint(gen_optimizer=gen_optimizer,
                                          disc_optimizer=disc_optimizer)

    ckpt_manager_models = tf.train.CheckpointManager(
        checkpoint=ckpt_models,
        directory=cfg.checkpoint_path + '/models/',
        max_to_keep=cfg.max_checkpoints_to_keep)

    ckpt_manager_optimizers = tf.train.CheckpointManager(
        checkpoint=ckpt_optimizers,
        directory=cfg.checkpoint_path + '/optimizers/',
        max_to_keep=cfg.max_checkpoints_to_keep)

    # Initialize log writer
    train_summary_writer = tf.summary.create_file_writer(cfg.log_dir)

    # Initialize metrics
    gen_loss = tf.keras.metrics.Metric
    start_time = time.time()
    num_images_before = 0
    num_minibatch = 0

    for example in dataset:

        disc_regularization = (num_minibatch % cfg.disc_reg_intervall == 0)
        gen_loss, disc_loss = train_step(real_images=example['data'],
                                         real_labels=example['label'],
                                         generator=generator,
                                         discriminator=discriminator,
                                         gen_optimizer=gen_optimizer,
                                         disc_optimizer=disc_optimizer,
                                         cfg=cfg,
                                         disc_regularization=disc_regularization)

        num_minibatch = gen_optimizer.iterations.numpy()
        num_images = num_minibatch * cfg.batch_size

        # Print Metrics
        if (num_images % (cfg.print_metrics_intervall_kimg * 1000)) < cfg.batch_size:
            images_per_second = (num_images - num_images_before) / (time.time() - start_time)
            print('minibatch {} images {} gen loss {:.4f} disc loss {:.4f}'
                  ' images per second {:.2f}'.format(num_minibatch,
                                                     num_images,
                                                     gen_loss,
                                                     disc_loss,
                                                     images_per_second))
            num_images_before = num_images
            start_time = time.time()

        # Save checkpoint
        if (num_images % (cfg.checkpoint_intervall_kimg * 1000)) < cfg.batch_size:
            save_checkpoint(ckpt_manager_models, num_images)
            save_checkpoint(ckpt_manager_optimizers, num_images)

        # Log metrics
        if (num_images % (cfg.log_metrics_intervall_kimg * 1000)) < cfg.batch_size:
            with train_summary_writer.as_default():
                tf.summary.scalar('gen_loss', gen_loss, step=num_images)
                tf.summary.scalar('disc_loss', disc_loss, step=num_images)

        if (num_images % (cfg.max_num_images_kimg * 1000)) < cfg.batch_size:
            # Save final state if not already done
            if not (num_images % cfg.checkpoint_intervall_kimg) < cfg.batch_size:
                save_checkpoint(ckpt_manager_models, int(num_images / 1000))
                save_checkpoint(ckpt_manager_optimizers, int(num_images / 1000))
            break
