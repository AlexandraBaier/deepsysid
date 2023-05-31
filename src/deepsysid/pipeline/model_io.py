import os
from typing import Callable

from ..models.base import DynamicIdentificationModel
from ..tracker.base import EventData


def load_model(
    model: DynamicIdentificationModel, directory: str, model_name: str
) -> None:
    extension = model.get_file_extension()
    model.load(
        tuple(os.path.join(directory, f'{model_name}.{ext}') for ext in extension)
    )


def save_model(
    model: DynamicIdentificationModel,
    directory: str,
    model_name: str,
    tracker: Callable[[EventData], None] = lambda _: None,
) -> None:
    extension = model.get_file_extension()
    model.save(
        tuple(os.path.join(directory, f'{model_name}.{ext}') for ext in extension),
        tracker,
    )
