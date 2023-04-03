import os
import pathlib

from deepsysid.cli.download import (
    TOY_DATASET_FOLDERNAMES_DICT,
    TOY_DATASET_ZIP_BASE_NAME,
    download_dataset_toy,
)


def test_download_dataset_toy(tmp_path: pathlib.Path) -> None:
    download_dataset_toy(target_directory=tmp_path)
    # check if path exist
    for rel_path in TOY_DATASET_FOLDERNAMES_DICT.values():
        tmp_dataset_path = os.path.join(tmp_path, TOY_DATASET_ZIP_BASE_NAME, rel_path)
        assert os.path.isdir(tmp_dataset_path)
