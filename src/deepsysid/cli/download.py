import io
import logging
import os
from typing import List

import pandas as pd
from pyDataverse.api import DataAccessApi, NativeApi

DARUS_BASE_URL = 'https://darus.uni-stuttgart.de/'
DOI_4_DOF_SIMULATED_SHIP = 'doi:10.18419/darus-2905'

logger = logging.getLogger(__name__)


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
        df = pd.read_csv(
            io.StringIO(  # type: ignore
                '\n'.join(map(lambda b: b.decode("utf-8"), response.iter_lines()))
            ),
            sep='\t',
        )
        df.to_csv(file_path, index=False)


def download_dataset_pelican_quadcopter(target_directory: str) -> None:
    pass
