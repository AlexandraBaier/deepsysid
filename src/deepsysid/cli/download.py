import io
import logging
import os
import random
import zipfile
from typing import List, Tuple

import numpy as np
import pandas as pd
import rarfile
import requests
import scipy.io
from numpy.typing import NDArray
from pyDataverse.api import DataAccessApi, NativeApi

DARUS_BASE_URL = 'https://darus.uni-stuttgart.de/'
DOI_4_DOF_SIMULATED_SHIP = 'doi:10.18419/darus-2905'

PELICAN_DOWNLOAD_URL = (
    'http://wavelab.uwaterloo.ca/wp-content/'
    'uploads/2017/09/AscTec_Pelican_Flight_Dataset.mat'
)
PELICAN_COLUMNS = ['Pos', 'Euler', 'Motors', 'Motors_CMD', 'Vel', 'pqr']
PELICAN_COLUMN_MAPPING = dict(
    Pos=['x', 'y', 'z'],
    Euler=['phi', 'theta', 'psi'],
    Motors=['n1', 'n2', 'n3', 'n4'],
    Motors_CMD=['n1_cmd', 'n2_cmd', 'n3_cmd', 'n4_cmd'],
    Vel=['dx', 'dy', 'dz'],
    pqr=['p', 'q', 'r'],
)
PELICAN_SAMPLING_TIME = 0.01

TOY_DATASETS_DOWNLOAD_URL = (
    'https://ipvs.informatik.uni-stuttgart.de/cloud/s'
    '/iswY6Q8wCqFLgji/download'
    '/ToyDatasets.zip'
)

TOY_DATASET_ZIP_BASE_NAME = 'ToyDataset'
TOY_DATASET_FOLDERNAMES_DICT = {
    'cartpole': 'cartpole/initial_state-0_K-200_T-30',
    'pendulum': 'pendulum/initial_state-0_K-200_T-30',
    'coupled-msd': 'coupled-msd/initial_state-0_K-200_T-1000',
}

INDUSTRIAL_ROBOT_DOWNLOAD_URL = (
    'https://fdm-fallback.uni-kl.de/TUK/FB/MV/WSKL/0001/'
    'Robot_Identification_Benchmark_Without_Raw_Data.rar'
)
INDUSTRIAL_ROBOT_SAMPLING_TIME = 0.1

F16_BENCHMARK_DOWNLOAD_URL = (
    'https://data.4tu.nl/file/b6dc643b-ecc6-437c-8a8a-1681650ec3fe/'
    '5414dfdc-6e8d-4208-be6e-fa553de9866f'
)
F16_INPUTS = ['Force']
F16_OUTPUTS = ['Acceleration1', 'Acceleration2', 'Acceleration3']

logger = logging.getLogger(__name__)


def download_dataset_toy(target_directory: str) -> None:

    logger.info(
        f'Downloading toy datasets from "{TOY_DATASETS_DOWNLOAD_URL}". '
        'This will take some time.'
    )
    target_directory = os.path.expanduser(target_directory)
    zip_path = os.path.join(target_directory, f'{TOY_DATASET_ZIP_BASE_NAME}.zip')
    response = requests.get(TOY_DATASETS_DOWNLOAD_URL, stream=True)

    with open(zip_path, mode='wb') as f:
        for chunk in response.iter_content(chunk_size=128):
            if chunk:
                f.write(chunk)

    with zipfile.ZipFile(zip_path, mode='r') as f:
        f.extractall(target_directory)


def download_dataset_4_dof_simulated_ship(
    routine_directory: str, ood_directory: str
) -> None:
    api = NativeApi(DARUS_BASE_URL)
    data_api = DataAccessApi(DARUS_BASE_URL)
    dataset = api.get_dataset(DOI_4_DOF_SIMULATED_SHIP)

    file_list: List[dict] = dataset.json()['data']['latestVersion']['files']
    for file in file_list:
        file_name = file['dataFile']['filename']
        file_id = file['dataFile']['id']

        if file.get('directoryLabel', None) is None:
            logger.info(f'Skipping {file_name}. Not a dataset file.')
            continue

        directory_root, *directory_elements = os.path.normpath(
            file['directoryLabel']
        ).split(os.sep)

        if directory_root == 'patrol_ship_routine':
            directory = os.path.expanduser(
                os.path.join(routine_directory, *directory_elements)
            )
        elif directory_root == 'patrol_ship_ood':
            directory = os.path.expanduser(
                os.path.join(ood_directory, *directory_elements)
            )
        else:
            logger.warning(
                f'Unexpected directory {directory_root} encountered. '
                'Does not match "patrol_ship_routine" or "patrol_ship_ood". '
                'Skipping.'
            )
            continue

        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, os.path.splitext(file_name)[0] + '.csv')
        logger.info(f'Downloading file to {file_path}.')
        response = data_api.get_datafile(file_id)

        def to_utf8(b: bytes) -> str:
            return b.decode('utf-8')

        response_io = io.StringIO('\n'.join(map(to_utf8, response.iter_lines())))
        df = pd.read_csv(
            response_io,
            sep='\t',
        )
        df.to_csv(file_path, index=False)


def download_dataset_pelican_quadcopter(
    target_directory: str, train_fraction: float, validation_fraction: float
) -> None:
    assert 0.0 < train_fraction < 1.0
    assert 0.0 < validation_fraction < 1.0
    assert (train_fraction + validation_fraction) < 1.0

    logger.info(
        f'Downloading Pelican dataset from "{PELICAN_DOWNLOAD_URL}". '
        'This will take some time.'
    )

    target_directory = os.path.expanduser(target_directory)
    raw_directory = os.path.join(target_directory, 'raw')
    interim_directory = os.path.join(target_directory, 'interim')
    processed_directory = os.path.join(target_directory, 'processed')
    os.makedirs(raw_directory, exist_ok=True)
    os.makedirs(interim_directory, exist_ok=True)
    os.makedirs(processed_directory, exist_ok=True)
    os.makedirs(os.path.join(processed_directory, 'train'), exist_ok=True)
    os.makedirs(os.path.join(processed_directory, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(processed_directory, 'test'), exist_ok=True)

    # https://stackoverflow.com/a/53101953
    response = requests.get(PELICAN_DOWNLOAD_URL, stream=True)
    with open(os.path.join(raw_directory, 'pelican.mat'), mode='wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    mat = scipy.io.loadmat(os.path.join(raw_directory, 'pelican.mat'))
    logger.info('Successfully finished download.')

    n_flights = mat['flights'].shape[1]
    total_steps = 0
    flights: List[pd.DataFrame] = []
    for flight_idx in range(n_flights):
        flight = mat['flights'][:, flight_idx][0]
        n_steps = flight['len'][0, 0][0, 0]
        total_steps += n_steps
        # Some features are computed and only n_steps - 1 features are available.
        time = np.arange(1, n_steps).astype(np.float64) * PELICAN_SAMPLING_TIME

        data: List[Tuple[str, NDArray[np.float64]]] = []
        for column in PELICAN_COLUMNS:
            for feature_idx, feature_name in enumerate(PELICAN_COLUMN_MAPPING[column]):
                feature = flight[column][0, 0][:, feature_idx]
                if feature.shape[0] == n_steps:
                    feature = feature[1:]
                data.append((feature_name, feature.astype(np.float64)))

        column_names = ['time'] + [column_name for column_name, _ in data]
        features = np.vstack([time] + [feature for _, feature in data]).T
        df = pd.DataFrame(data=features, columns=column_names)
        flights.append(df)

    random.seed(12345)
    flight_indexes = list(range(n_flights))
    random.shuffle(flight_indexes)

    # Just shuffling files and selecting fraction of files is not sufficient,
    # since files contain different length sequences.
    # Instead, we shuffle and then fill the train set and test set
    # until they contain enough samples. Validation set will end up smaller
    # than specified but is the most forgivable to get wrong.
    test_fraction = 1.0 - (train_fraction + validation_fraction)
    train_cutoff = 0
    train_steps = 0
    for flight_idx in flight_indexes:
        train_cutoff += 1
        train_steps += flights[flight_idx].shape[0]
        if train_fraction < (train_steps / total_steps):
            break

    test_cutoff = train_cutoff
    test_steps = 0
    for flight_idx in flight_indexes[train_cutoff:]:
        test_cutoff += 1
        test_steps += flights[flight_idx].shape[0]
        if test_fraction < (test_steps / total_steps):
            break

    train_set = set(flight_indexes[:train_cutoff])
    test_set = set(flight_indexes[train_cutoff:test_cutoff])

    for flight_idx, flight in enumerate(flights):
        if flight_idx in train_set:
            set_type = 'train'
        elif flight_idx in test_set:
            set_type = 'test'
        else:
            set_type = 'validation'

        interim_path = os.path.join(interim_directory, f'{flight_idx}.csv')
        processed_path = os.path.join(
            processed_directory, set_type, f'{flight_idx}.csv'
        )

        flight.to_csv(interim_path, index=False)
        flight.to_csv(processed_path, index=False)

    logger.info(
        f'Train%: {train_steps / total_steps:.2%}, '
        f'Test%: {test_steps / total_steps:.2%}'
    )
    logger.info('Finished Pelican dataset download and preparation.')


def download_dataset_industrial_robot(
    target_directory: str, validation_fraction: float
) -> None:
    if validation_fraction < 0.0:
        raise ValueError('Validation fraction cannot be smaller than 0.')
    if validation_fraction > 1.0:
        raise ValueError('Validation fraction cannot be larger than 1.')

    logger.info(
        f'Downloading industrial robot dataset from "{INDUSTRIAL_ROBOT_DOWNLOAD_URL}". '
        'This will take some time.'
    )

    target_directory = os.path.expanduser(target_directory)
    raw_directory = os.path.join(target_directory, 'raw')
    processed_directory = os.path.join(target_directory, 'processed')
    os.makedirs(raw_directory, exist_ok=True)
    os.makedirs(processed_directory, exist_ok=True)
    os.makedirs(os.path.join(processed_directory, 'train'), exist_ok=True)
    os.makedirs(os.path.join(processed_directory, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(processed_directory, 'test'), exist_ok=True)

    # https://stackoverflow.com/a/53101953
    rar_path = os.path.join(raw_directory, 'industrial_robot.zip')
    mat_path = os.path.join(raw_directory, 'industrial_robot.mat')
    response = requests.get(INDUSTRIAL_ROBOT_DOWNLOAD_URL, stream=True)
    with open(rar_path, mode='wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

    with rarfile.RarFile(rar_path) as f:
        with f.open(
            'forward_identification_without_raw_data.mat', mode='r'
        ) as mat_from_rar:
            with open(mat_path, mode='wb') as target_mat:
                target_mat.write(mat_from_rar.read())

    mat = scipy.io.loadmat(mat_path)
    logger.info('Successfully finished download.')

    u_train_val = mat['u_train'].T
    y_train_val = mat['y_train'].T
    u_test = mat['u_test'].T
    y_test = mat['y_test'].T

    n_train_val = u_train_val.shape[0]
    n_train = int(n_train_val * (1.0 - validation_fraction))

    u_val = u_train_val[n_train:]
    y_val = u_train_val[n_train:]

    u_train = u_train_val[:n_train]
    y_train = y_train_val[:n_train]

    time_train = (
        np.arange(u_train.shape[0])[:, np.newaxis] * INDUSTRIAL_ROBOT_SAMPLING_TIME
    )
    time_val = np.arange(u_val.shape[0])[:, np.newaxis] * INDUSTRIAL_ROBOT_SAMPLING_TIME
    time_test = (
        np.arange(u_test.shape[0])[:, np.newaxis] * INDUSTRIAL_ROBOT_SAMPLING_TIME
    )

    tuy_train = np.hstack((time_train, u_train, y_train))
    tuy_val = np.hstack((time_val, u_val, y_val))
    tuy_test = np.hstack((time_test, u_test, y_test))

    input_dim = u_train.shape[1]
    output_dim = y_train.shape[1]

    for dataset, mode in (
        (tuy_train, 'train'),
        (tuy_val, 'validation'),
        (tuy_test, 'test'),
    ):
        processed_path = os.path.join(
            processed_directory, mode, f'robot_arm_{mode}.csv'
        )
        df = pd.DataFrame(
            data=dataset,
            columns=(
                ['time']
                + [f'u{i}' for i in range(input_dim)]
                + [f'y{i}' for i in range(output_dim)]
            ),
        )
        df.to_csv(processed_path, index=False)
    logger.info('Finished Industrial Robot dataset download and preparation.')

def download_dataset_f16_aircraft(target_directory: str, validation_fraction:float) -> None:
    if validation_fraction < 0.0:
        raise ValueError('Validation fraction cannot be smaller than 0.')
    if validation_fraction > 1.0:
        raise ValueError('Validation fraction cannot be larger than 1.')

    logger.info(
        f'Downloading F16 GVT Benchmark dataset from "{F16_BENCHMARK_DOWNLOAD_URL}". '
        'This will take some time.'
    )

    target_directory = os.path.expanduser(target_directory)
    raw_directory = os.path.join(target_directory, 'raw')
    processed_directory = os.path.join(target_directory, 'processed')
    os.makedirs(raw_directory, exist_ok=True)
    os.makedirs(processed_directory, exist_ok=True)
    os.makedirs(os.path.join(processed_directory, 'train'), exist_ok=True)
    os.makedirs(os.path.join(processed_directory, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(processed_directory, 'test'), exist_ok=True)

    # https://stackoverflow.com/a/53101953
    zip_path = os.path.join(raw_directory, 'F16-Ground-Vibration-Test.zip')
    response = requests.get(F16_BENCHMARK_DOWNLOAD_URL, stream=True) 
    with open(zip_path, mode='wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

    base_path = os.path.join('F16GVT_Files', 'BenchmarkData')
    base_name = 'F16Data'
    excitation_type = 'FullMSine'
    training_levels = ['1','3','5','7']
    test_levels = ['2','4','6']
    training_file_names = [f'{base_name}_{excitation_type}_Level{level}.csv' for level in training_levels]
    test_file_names = [f'{base_name}_{excitation_type}_Level{level}_Validation.csv' for level in test_levels]
    with zipfile.ZipFile(zip_path) as f:
        for file_name in training_file_names + test_file_names:
            with f.open(os.path.join(base_path, file_name), mode='r') as csv_from_zip:
                with open(os.path.join(raw_directory, file_name), mode='wb') as target_csv:
                    target_csv.write(csv_from_zip.read())

    logger.info('Successfully finished download.')
    test_file_paths = [os.path.join(raw_directory, file_name) for file_name in test_file_names]

    input_dim = len(F16_INPUTS)
    output_dim = len(F16_OUTPUTS)
    for idx, test_file in enumerate(test_file_paths):
        processed_path = os.path.join(processed_directory, 'test',f'F16-Ground-Vibration-Test_test-{idx}.csv')
        df = pd.read_csv(test_file)
        N = df.shape[0]
        time_test = np.linspace(0,(N-1)*1/df['Fs'][0], N).reshape(N,1)
        u_test = np.array(df[F16_INPUTS]).reshape(N,input_dim)
        y_test = np.array(df[F16_OUTPUTS]).reshape(N,output_dim)
        tuy_test = np.hstack((time_test, u_test, y_test))
        df_out = pd.DataFrame(
            data=tuy_test,
            columns=['time']+['F'] + [f'a{i}' for i in range(output_dim)]
        )
        df_out.to_csv(processed_path, index=False)

    for idx, train_file in enumerate([os.path.join(raw_directory, file_name) for file_name in training_file_names]):
        df = pd.read_csv(train_file)

        N_train_val = df.shape[0]
        u_train_val = np.array(df[F16_INPUTS]).reshape(N_train_val, input_dim)
        y_train_val = np.array(df[F16_OUTPUTS]).reshape(N_train_val, output_dim)

        N_train = int(N_train_val * (1.0 - validation_fraction))
        t_train = np.linspace(0,(N_train-1)*1/df['Fs'][0], N_train).reshape(N_train,1)
        u_train = u_train_val[:N_train, :]
        y_train = y_train_val[:N_train, :]
        tuy_train = np.hstack((t_train, u_train, y_train))

        N_val = N_train_val - N_train
        t_val = np.linspace(0,(N_val-1)*1/df['Fs'][0], N_val).reshape(N_val,1)
        u_val = u_train_val[N_train:, :]
        y_val = y_train_val[N_train:, :]
        tuy_val = np.hstack((t_val, u_val, y_val))

        for dataset, mode in ((tuy_train, 'train'), (tuy_val,'validation')):
            processed_path = os.path.join(processed_directory, mode,f'F16-Ground-Vibration-Test_{mode}-{idx}.csv')

            df_out = pd.DataFrame(
                data=dataset,
                columns=['time']+['F'] + [f'a{i}' for i in range(output_dim)]
            )
            df_out.to_csv(processed_path, index=False)
    logger.info('Finished F16 Ground Vibration Test dataset download and preparation.')