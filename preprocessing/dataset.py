import os
from os.path import isfile, join

import PIL
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from model.utils import ModelConfig
from preprocessing.utils import plot_batch


def crop_to_rectangular(image):
    """ Crops longer dimension to make image rectangular
    Input: Array in [HWC] or [HW]
    Output: [DDC] or [DD] with D = H if H < W and D = W otherwise.
    """

    input_shape = image.shape
    if len(input_shape) == 2:
        image = image[:, :, np.newaxis]
    delta = image.shape[1]-image.shape[0]
    if delta > 0:
        cropped_img = image[:, (delta // 2): - (delta // 2) - (delta % 2), :]
    elif delta < 0:
        cropped_img = image[(-delta // 2): - (-delta // 2) - (-delta % 2), :, :]
    else:
        cropped_img = image

    if len(input_shape) == 2:
        cropped_img = cropped_img[:, :, 0]
    return cropped_img


def preprocess_images(input_dir, out_dir, resolution, to_grayscale=False):
    image_file_names = [f for f in os.listdir(input_dir)
                        if os.path.isfile(join(input_dir, f))]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for image_file_name in image_file_names:
        image = Image.open(os.path.join(input_dir, image_file_name))
        if to_grayscale:
            image = image.convert('L')
        image_array = np.asarray(image)
        cropped_img = crop_to_rectangular(image_array)
        final_img = Image.fromarray(cropped_img).resize((resolution, resolution), Image.ANTIALIAS)
        final_img.save(os.path.join(out_dir, image_file_name))


def prepare_mnist_images(mnist_dir):
    with open(os.path.join(mnist_dir, 'train-images-idx3-ubyte'), 'rb') as file:
        images = np.frombuffer(file.read(), np.uint8, offset=16)

    with open(os.path.join(mnist_dir, 'train-labels-idx1-ubyte'), 'rb') as file:
        labels = np.frombuffer(file.read(), np.uint8, offset=8)

    images = images.reshape(-1, 28, 28, 1)
    images = np.pad(images,
                    pad_width=[(0, 0), (2, 2), (2, 2), (0, 0)],
                    mode='constant',
                    constant_values=0)

    labels_one_hot = np.zeros((labels.size, 10), dtype=np.float32)
    labels_one_hot[np.arange(labels.size), labels] = 1.0

    return images, labels_one_hot


def save_image_to_png(image, path):
    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.savefig(path)
    plt.close()


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def image_example(image, label):
    feature = {
        'data': _bytes_feature(image.tostring()),
        'label': _bytes_feature(label.tostring())
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_images_to_tf_record(images, labels, tf_record_file_path):
    with tf.io.TFRecordWriter(tf_record_file_path) as writer:
        for image, label in zip(images, labels):
            tf_example = image_example(image, label)
            writer.write(tf_example.SerializeToString())


def _parse_image_function(example_proto, cfg):
    image_feature_description = {'data': tf.io.FixedLenFeature([], tf.string),
                                 'label': tf.io.FixedLenFeature([], tf.string)}

    example = {}
    parsed_example = tf.io.parse_single_example(example_proto, image_feature_description)
    decoded_example_data = tf.io.decode_raw(parsed_example['data'], out_type=tf.uint8)
    if cfg.label_conditioning:
        decoded_example_label = tf.io.decode_raw(parsed_example['label'], out_type=tf.float32)
        example['label'] = tf.dtypes.cast(tf.reshape(
            decoded_example_label, [cfg.labels_size]), tf.float32)
    else:
        example['label'] = None

    example['data'] = tf.dtypes.cast(tf.reshape(
        decoded_example_data, [cfg.resolution, cfg.resolution, cfg.num_channels]), tf.float32)

    return example


def read_dataset(dataset_path, cfg):
    raw_dataset = tf.data.TFRecordDataset(dataset_path)

    dataset = (raw_dataset
               .map(
                   lambda example_proto: _parse_image_function(example_proto, cfg)
               )
               .shuffle(buffer_size=cfg.shuffle_buffer_size)
               .repeat()
               .batch(cfg.batch_size))

    return dataset


def prepare_imagenet_sketch_images(config,
                                   raw_data_path,
                                   data_out_path,
                                   to_grayscale=True):

    image_file_dirs = [f for f in os.listdir(raw_data_path)
                       if os.path.isdir(join(raw_data_path, f))]

    for image_file_dir in image_file_dirs:
        preprocess_images(input_dir=join(raw_data_path, image_file_dir),
                          out_dir=join(data_out_path, 'preprocessed_images', image_file_dir),
                          resolution=config.resolution,
                          to_grayscale=config.num_channels == 1)


def prepare_flower_images(config,
                          raw_data_path,
                          data_out_path,
                          to_grayscale=True):
    preprocess_images(input_dir=raw_data_path,
                      out_dir=join(data_out_path, 'preprocessed_images'),
                      resolution=config.resolution,
                      to_grayscale=config.num_channels == 1)


def convert_images_to_tf_record(preprocess_images_path, tf_record_file_path):

    image_file_dirs = [f for f in os.listdir(preprocess_images_path)
                       if os.path.isdir(join(preprocess_images_path, f))]

    with tf.io.TFRecordWriter(tf_record_file_path) as writer:
        for image_file_dir in image_file_dirs:

            label_one_hot = np.zeros((len(image_file_dirs)), dtype=np.float32)
            label_one_hot[image_file_dirs.index(image_file_dir)] = 1.0

            image_file_paths = [f for f in os.listdir(join(preprocess_images_path, image_file_dir))
                                if os.path.isfile(join(preprocess_images_path, image_file_dir, f))]

            for image_file_path in image_file_paths:
                image = np.asarray(Image.open(join(preprocess_images_path, image_file_dir, image_file_path)))

                tf_example = image_example(image, label_one_hot)
                writer.write(tf_example.SerializeToString())


def convert_flower_images_to_tf_record(preprocess_images_path, tf_record_file_path):

    with tf.io.TFRecordWriter(tf_record_file_path) as writer:

        image_file_paths = [f for f in os.listdir(preprocess_images_path)
                            if os.path.isfile(join(preprocess_images_path, f))]

        for image_file_path in image_file_paths:
            image = np.asarray(Image.open(join(preprocess_images_path, image_file_path)))

            tf_example = image_example(image, np.array([1]))
            writer.write(tf_example.SerializeToString())


def preprocess_data(config_path, dataset, raw_data_path, data_out_path):
    dataset_prep = {'mnist': preprocess_mnist,
                    'imagenet_sketch': preprocess_imagenet_sketch,
                    'flowers': preprocess_flowers}
    dataset_prep[dataset](config_path, raw_data_path, data_out_path)


def preprocess_imagenet_sketch(config_path, raw_data_path, data_out_path):
    cfg = ModelConfig(config_path)
    prepare_imagenet_sketch_images(cfg, raw_data_path, data_out_path)
    convert_images_to_tf_record(join(data_out_path, 'preprocessed_images'),
                                join(data_out_path, 'imagenet_sketch.tfrecords'))


def preprocess_flowers(config_path, raw_data_path, data_out_path):
    cfg = ModelConfig(config_path)
    prepare_flower_images(cfg, raw_data_path, data_out_path)
    convert_flower_images_to_tf_record(join(data_out_path, 'preprocessed_images'),
                                       join(data_out_path, 'flowers.tfrecords'))


def preprocess_mnist(config_path, dataset, raw_data_path, data_out_path):
    images, labels = prepare_mnist_images(raw_data_path)
    write_images_to_tf_record(images, labels, data_out_path)


def plot_train_images(data_path, config_path, num_batches, out_path):
    cfg = ModelConfig(config_path)
    dataset = read_dataset(data_path, cfg)
    for n, batch in enumerate(dataset.take(num_batches)):
        plot_batch(batch['data'], batch['label'], join(out_path, 'batch-{}.png'.format(n + 1)), cfg.label_conditioning)
