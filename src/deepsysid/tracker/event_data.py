import dataclasses
from typing import Any, Dict, Optional, Union

from ..models.utils import TrainingPrediction
from .configuration import ExperimentTrackingConfiguration


@dataclasses.dataclass
class EventData:
    msg: str


@dataclasses.dataclass
class StopRun(EventData):
    run_status: Optional[str]


@dataclasses.dataclass
class EventReturn:
    data: Dict[str, Any]


@dataclasses.dataclass
class TrackParameters(EventData):
    parameters: Dict[str, Union[str, float, int]]


@dataclasses.dataclass
class TrackFigures(EventData):
    results: TrainingPrediction
    name: str


@dataclasses.dataclass
class TrackArtifacts(EventData):
    artifacts: Dict[str, str]


@dataclasses.dataclass
class SetExperiment(EventData):
    dataset_directory: str


@dataclasses.dataclass
class SaveTrackingConfiguration(EventData):
    config: Dict[str, ExperimentTrackingConfiguration]
    model_name: str
    model_directory: str


@dataclasses.dataclass
class LoadTrackingConfiguration(EventData):
    model_directory: str
    model_name: str


@dataclasses.dataclass
class SetTags(EventData):
    tags: Dict[str, Union[str, bool]]


@dataclasses.dataclass
class TrackMetrics(EventData):
    metrics: Dict[str, float]
