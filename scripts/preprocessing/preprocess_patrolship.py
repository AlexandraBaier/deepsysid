import json
import os
import random

import pandas as pd


def get_subdirectories(directory):
    return [
        item
        for item in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, item)) and item not in {'.', '..'}
    ]


def main():
    root_dir = os.environ['DATASET_DIRECTORY']
    raw_dir = os.path.join(root_dir, 'raw')
    interim_dir = os.path.join(root_dir, 'interim')
    processed_dir = os.path.join(root_dir, 'processed')

    try:
        os.mkdir(interim_dir)
    except FileExistsError:
        pass
    try:
        os.mkdir(processed_dir)
    except FileExistsError:
        pass

    with open(os.environ['CONFIGURATION'], mode='r') as f:
        config = json.load(f)

    state_names = config['state_names']
    control_names = config['control_names']

    simulation_names = get_subdirectories(raw_dir)
    measured_sim_df = dict()
    true_sim_df = dict()
    for simulation_name in simulation_names:
        measured_df = pd.read_csv(
            os.path.join(interim_dir, simulation_name, 'measurements.csv')
        )
        measured_df = measured_df.rename(columns={'nr': 'n'})
        measured_df = measured_df.loc[:, ['time'] + control_names + state_names]
        measured_sim_df[simulation_name] = measured_df

        true_df = pd.read_csv(
            os.path.join(interim_dir, simulation_name, 'true_measurements.csv')
        )
        true_df = true_df.rename(columns={'nr': 'n'})
        true_df = true_df.loc[:, ['time'] + control_names + state_names]
        true_sim_df[simulation_name] = true_df

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
        measured_sim_df[simulation_name].to_csv(
            os.path.join(processed_dir, 'train', simulation_name + '.csv'), index=False
        )
    # use data without sensor noise for validation
    for simulation_name in validation_simulations:
        true_sim_df[simulation_name].to_csv(
            os.path.join(processed_dir, 'validation', simulation_name + '.csv'),
            index=False,
        )
    # but use data without sensor noise for testing
    for simulation_name in test_simulations:
        true_sim_df[simulation_name].to_csv(
            os.path.join(processed_dir, 'test', simulation_name + '.csv'), index=False
        )


if __name__ == '__main__':
    main()
