import PIL.Image
import PIL.ImageFont

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Agg')


def plot_batch(fake_images, fake_labels, file_path, plot_labels):

    if plot_labels:
        fake_labels_dense = np.argmax(fake_labels, axis=1)
    grid_size = np.ceil(np.sqrt(fake_images.shape[0]))
    fig = plt.figure(figsize=(grid_size, grid_size))

    for i in range(fake_images.shape[0]):
        ax = fig.add_subplot(grid_size, grid_size, i + 1)
        if plot_labels:
            ax.title.set_text('{}'.format(fake_labels_dense[i]))
        img_data = fake_images[i, :, :, :] if fake_images.shape[3] > 1 else fake_images[i, :, :, 0]
        image = np.rint(img_data).clip(0, 255).astype(np.uint8)
        color_map = None if fake_images.shape[3] > 1 else 'gray'
        plt.imshow(image, cmap=color_map)

        plt.axis('off')

    plt.savefig(file_path)
    plt.close()
