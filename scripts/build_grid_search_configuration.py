import argparse
import json
import os

from deepsysid.execution import ExperimentGridSearchTemplate, ExperimentConfiguration


def main():
    parser = argparse.ArgumentParser(
        description='Build configuration file given grid-search template.'
    )
    parser.add_argument('template', help='path to grid-search template')
    args = parser.parse_args()

    with open(os.path.expanduser(args.template), mode='r') as f:
        template = json.load(f)

    grid_search_template = ExperimentGridSearchTemplate.parse_obj(template)
    configuration = ExperimentConfiguration.from_grid_search_template(
        grid_search_template, 'cpu'
    )

    with open(os.environ['CONFIGURATION'], mode='w') as f:
        json.dump(configuration.dict(), f)


if __name__ == '__main__':
    main()
