import argparse
import itertools
import json
import os


def main():
    parser = argparse.ArgumentParser(description='Build configuration file given grid-search template.')
    parser.add_argument('template', help='path to grid-search template')
    args = parser.parse_args()

    with open(os.path.expanduser(args.template), mode='r') as f:
        template = json.load(f)

    config = dict()
    config['train_fraction'] = template['settings']['train_fraction']
    if 'validation_fraction' in template['settings']:
        config['validation_fraction'] = template['settings']['validation_fraction']
    config['time_delta'] = template['settings']['time_delta']
    config['window'] = template['settings']['window']
    config['horizon'] = template['settings']['horizon']
    config['control_names'] = template['settings']['control_names']
    config['state_names'] = template['settings']['state_names']
    config['thresholds'] = template['settings']['thresholds']

    config['models'] = dict()
    for model_template in template['models']:
        flexible_params = list(model_template['flexible_parameters'].items())
        flexible_param_names = [name for name, _ in flexible_params]
        for combination in itertools.product(*[param_values for _, param_values in flexible_params]):
            name_parts = list()
            model_config = dict()
            model_config['parameters'] = dict()
            # static parameters
            for name, value in model_template['static_parameters'].items():
                model_config['parameters'][name] = value
            # flexible parameters
            for name, value in zip(flexible_param_names, combination):
                model_config['parameters'][name] = value
                if issubclass(type(value), list):
                    name_parts.append('_'.join([str(v) for v in value]))
                else:
                    name_parts.append(str(value).replace('.', ''))
            # meta parameters
            model_config['model_class'] = model_template['model_class']
            model_name = model_template['model_base_name'] + ('-' if name_parts else '') + '-'.join(name_parts)
            model_config['location'] = os.path.join(template['base_path'], model_name)
            config['models'][model_name] = model_config

    with open(os.environ['CONFIGURATION'], mode='w') as f:
        json.dump(config, f)


if __name__ == '__main__':
    main()
