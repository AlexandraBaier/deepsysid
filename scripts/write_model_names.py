import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description='Write all model names from configuration.')
    parser.add_argument('output')
    args = parser.parse_args()

    with open(os.environ['CONFIGURATION'], mode='r') as f:
        config = json.load(f)

    model_names = list(config['models'])

    with open(args.output, mode='w') as f:
        f.write('\n'.join(model_names) + '\n')


if __name__ == '__main__':
    main()
