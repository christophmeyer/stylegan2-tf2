import argparse

from model.training import train_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='run_training')
    parser.add_argument('--config_path', help='path of the config file')
    parser.add_argument('--data_path', help='path of the preprocessed training data')
    args = parser.parse_args()

    train_model(config_path=args.config_path,
                data_path=args.data_path)
