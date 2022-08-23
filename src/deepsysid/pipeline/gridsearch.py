from typing import Dict, List

from pydantic import BaseModel

from .configuration import ExperimentConfiguration, ExperimentGridSearchTemplate


class EvaluationScore:
    metric: str
    variable: str
    score: float


class GridSearchSession(BaseModel):
    remaining_models: List[str]
    validated_models: List[str]
    validation_complete: bool = False
    validation_scores: Dict[str, List[EvaluationScore]]
    best_model_per_class: Dict[str, str]
    best_model_per_group: Dict[str, str]
    test_scores: Dict[str, List[EvaluationScore]]


def gridsearch(template: ExperimentGridSearchTemplate, device_name: str) -> None:
    config = ExperimentConfiguration.from_grid_search_template(template, device_name)
    print(config)
