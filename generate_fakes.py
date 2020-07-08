import argparse

from postprocessing.generation import generate_fakes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='generate_fakes')
    parser.add_argument('--config_path', help='path of the config file')
    parser.add_argument('--num_fake_batches', type=int, help='number of fake batches to generate')
    parser.add_argument('--checkpoint_dir', help='checkpoint dir of the generator network')
    parser.add_argument('--generated_images_dir', help='dir for the generated images')
    args = parser.parse_args()

    generate_fakes(config_path=args.config_path,
                   num_fake_batches=args.num_fake_batches,
                   checkpoint_dir=args.checkpoint_dir,
                   generated_images_dir=args.generated_images_dir)
