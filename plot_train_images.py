import argparse

from preprocessing.dataset import plot_train_images

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='generate_fakes')
    parser.add_argument('--data_path', help='path of the tfrecords train data file')
    parser.add_argument('--config_path', help='path of the config file')
    parser.add_argument('--num_batches', type=int, help='number of fake batches to generate')
    parser.add_argument('--out_path', help='checkpoint dir of the generator network')
    args = parser.parse_args()

    plot_train_images(data_path=args.data_path,
                      config_path=args.config_path,
                      num_batches=args.num_batches,
                      out_path=args.out_path)
