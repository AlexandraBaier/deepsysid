import json
import os
import random

import pandas as pd


def main() -> None:
    root_dir = os.path.expanduser(os.environ['DATASET_DIRECTORY'])
    raw_dir = os.path.join(root_dir, 'raw')
    processed_dir = os.path.join(root_dir, 'processed')

    try:
        os.mkdir(processed_dir)
    except FileExistsError:
        pass

    with open(os.path.expanduser(os.environ['CONFIGURATION']), mode='r') as f:
        config = json.load(f)

    state_names = config['state_names']
    control_names = config['control_names']

    simulation_names = [
        fn for fn in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, fn))
    ]
    sim_df = dict()
    for simulation_name in simulation_names:
        df = pd.read_csv(os.path.join(raw_dir, simulation_name))
        df = df.loc[:, control_names + state_names]
        sim_df[simulation_name] = df

    try:
        os.mkdir(os.path.join(processed_dir, 'train'))
    except FileExistsError:
        pass
    try:
        os.mkdir(os.path.join(processed_dir, 'test'))
    except FileExistsError:
        pass
    if config.get('validation_fraction', None):
        try:
            os.mkdir(os.path.join(processed_dir, 'validation'))
        except FileExistsError:
            pass

    train_fraction = config['train_fraction']
    train_count = int(train_fraction * len(simulation_names))

    validation_fraction = config.get('validation_fraction', 0.0)
    validation_count = int(validation_fraction * len(simulation_names))

    random.seed(1337)
    random.shuffle(simulation_names)

    train_simulations = simulation_names[:train_count]
    validation_simulations = simulation_names[
        train_count : train_count + validation_count
    ]
    test_simulations = simulation_names[train_count + validation_count :]

    # use data with sensor noise for training
    for simulation_name in train_simulations:
        sim_df[simulation_name].to_csv(
            os.path.join(processed_dir, 'train', simulation_name), index=False
        )
    # use data without sensor noise for validation
    for simulation_name in validation_simulations:
        sim_df[simulation_name].to_csv(
            os.path.join(processed_dir, 'validation', simulation_name), index=False
        )
    # but use data without sensor noise for testing
    for simulation_name in test_simulations:
        sim_df[simulation_name].to_csv(
            os.path.join(processed_dir, 'test', simulation_name), index=False
        )


if __name__ == '__main__':
    main()
