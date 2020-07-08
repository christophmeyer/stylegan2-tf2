import argparse

from preprocessing.dataset import preprocess_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='preprocess_data')
    parser.add_argument('--config_path', help='path of the config file')
    parser.add_argument('--dataset', help='name of the dataset')
    parser.add_argument('--raw_data_path', help='path of the raw image data')
    parser.add_argument('--data_out_path', help='path for the preprocessed training data')
    args = parser.parse_args()

    preprocess_data(args.config_path, args.dataset, args.raw_data_path, args.data_out_path)
