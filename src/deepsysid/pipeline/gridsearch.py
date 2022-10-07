import json
import logging
import os.path
from enum import Enum
from typing import Callable, Dict, Literal, Optional, Set

from pydantic import BaseModel

from .configuration import ExperimentConfiguration, ExperimentGridSearchTemplate
from .data_io import build_score_file_name
from .evaluation import ReadableEvaluationScores, evaluate_model
from .testing import test_model
from .training import train_model

logger = logging.getLogger(__name__)


class ExperimentSessionReport(BaseModel):
    unfinished_models: Set[str]
    validated_models: Set[str]
    tested_models: Optional[Set[str]] = None
    best_per_class: Optional[Dict[str, str]] = None
    best_per_base_name: Optional[Dict[str, str]] = None


class ConfigurationSessionDifference(BaseModel):
    new_to_configuration: Set[str]
    unfinished_missing_from_configuration: Set[str]
    validated_missing_from_configuration: Set[str]


class SessionAction(Enum):
    NEW = 0
    CONTINUE = 1
    TEST_BEST = 2


class ExperimentSessionManager(object):
    def __init__(
        self,
        config: ExperimentGridSearchTemplate,
        device_name: str,
        session_action: SessionAction,
        dataset_directory: str,
        models_directory: str,
        results_directory: str,
        session_report: Optional[ExperimentSessionReport] = None,
    ) -> None:
        self.config = config
        self.device_name = device_name
        self.experiment_config = ExperimentConfiguration.from_grid_search_template(
            config, device_name=self.device_name
        )
        self.session_action = session_action
        self.dataset_directory = dataset_directory
        self.models_directory = models_directory
        self.results_directory = results_directory

        if session_action == SessionAction.NEW and session_report is not None:
            raise ValueError('Cannot start New session with existing session report.')

        if session_action == SessionAction.CONTINUE:
            if session_report is None:
                raise ValueError('Cannot Continue without session report..')
            else:
                session_report = self._validate_session_report(session_report)

        if session_action == SessionAction.TEST_BEST:
            if session_report is None:
                raise ValueError('Cannot Test Best models without session report.')
            else:
                session_report = self._validate_session_report(session_report)
                if len(session_report.unfinished_models) > 0:
                    raise ValueError(
                        'Found following unfinished models in session report. '
                        'Cannot test best performing models '
                        'while not all models are validated:'
                        f'{", ".join(session_report.unfinished_models)}.'
                    )

        if session_report is None:
            self.session_report: ExperimentSessionReport = ExperimentSessionReport(
                unfinished_models=set(self.experiment_config.models.keys()),
                validated_models=set(),
            )
        else:
            self.session_report = session_report.copy(deep=True)

    def get_session_report(self) -> ExperimentSessionReport:
        return self.session_report.copy(deep=True)

    def run_session(
        self, callback: Callable[[ExperimentSessionReport], None] = lambda _: None
    ) -> None:
        if self.session_action in {SessionAction.NEW, SessionAction.CONTINUE}:
            self._run_validation(callback)
        elif self.session_action == SessionAction.TEST_BEST:
            self._run_test_best()
        else:
            raise NotImplementedError(
                f'{self.session_action} has no implemented functionality.'
            )

    def _run_validation(
        self, callback: Callable[[ExperimentSessionReport], None]
    ) -> ExperimentSessionReport:
        logger.info(
            f'Found {len(self.session_report.unfinished_models)} unfinished models '
            f'and {len(self.session_report.validated_models)} validated models.'
        )
        unfinished_models = self.session_report.unfinished_models.copy()
        for model_name in unfinished_models:
            train_model(
                model_name=model_name,
                device_name=self.device_name,
                configuration=self.experiment_config,
                dataset_directory=self.dataset_directory,
                models_directory=self.models_directory,
            )
            self._run_test_eval(model_name=model_name, mode='validation')
            self.session_report = self._update_from_unfinished_to_validated(model_name)

            callback(self.get_session_report())

            n_unfinished_models = len(self.session_report.unfinished_models)
            n_total_models = n_unfinished_models + len(
                self.session_report.validated_models
            )
            logger.info(f'Trained and validated: {model_name}.')
            logger.info(
                'Validation Progress: '
                f'{n_total_models - n_unfinished_models}/{n_total_models}.'
            )
        return self.get_session_report()

    def _run_test_best(self) -> ExperimentSessionReport:
        model2score = dict(
            (model_name, self._get_validation_score(model_name))
            for model_name in self.session_report.validated_models
        )
        model2class = dict(
            (model_name, self.experiment_config.models[model_name].model_class)
            for model_name in self.session_report.validated_models
        )
        model_classes = set(model2class.values())

        base_names = set(
            model_template.model_base_name for model_template in self.config.models
        )
        model2base_name = dict()
        for model_name in self.session_report.validated_models:
            valid_base_names = []
            for base_name in base_names:
                if model_name[: len(base_name)] == base_name:
                    valid_base_names.append(base_name)
            # Longest matching model_base_name is the correct base name.
            # Every model is guaranteed to have at least one match
            # due to grid search model name generation.
            model2base_name[model_name] = sorted(
                valid_base_names, key=lambda bn: len(bn), reverse=True
            )[0]

        best_per_class = dict(
            (
                model_class,
                min(
                    # Iterate over all models with matching class
                    # to select model with the lowest score.
                    (
                        (model_name, model2score[model_name])
                        for model_name, model_score in model2score.items()
                        if model2class[model_name] == model_class
                    ),
                    key=lambda t: t[1],
                )[0],
            )
            for model_class in model_classes
        )
        best_per_base_name = dict(
            (
                base_name,
                min(
                    # Iterate over all models with matching base name
                    # to select model with the lowest score.
                    (
                        (model_name, model2score[model_name])
                        for model_name, model_score in model2score.items()
                        if model2base_name[model_name] == base_name
                    ),
                    key=lambda t: t[1],
                )[0],
            )
            for base_name in base_names
        )
        models_to_test = set(best_per_class.values()).union(best_per_base_name.values())
        for model_name in models_to_test:
            self._run_test_eval(model_name=model_name, mode='test')
            logger.info(f'Tested: {model_name}')

        self.session_report = ExperimentSessionReport(
            unfinished_models=self.session_report.unfinished_models.copy(),
            validated_models=self.session_report.validated_models.copy(),
            tested_models=models_to_test,
            best_per_class=best_per_class,
            best_per_base_name=best_per_base_name,
        )

        return self.get_session_report()

    def _get_validation_score(self, model_name: str) -> float:
        score_file_name = build_score_file_name(
            mode='validation',
            window_size=self.config.settings.window_size,
            horizon_size=self.config.settings.horizon_size,
            extension='json',
            threshold=None,
        )
        with open(
            os.path.join(self.results_directory, model_name, score_file_name)
        ) as f:
            scores: ReadableEvaluationScores = ReadableEvaluationScores.parse_obj(
                json.load(f)
            )
        validation_score = sum(
            scores.scores_per_horizon[self.config.settings.horizon_size][
                self.experiment_config.target_metric
            ]
        )

        return validation_score

    def _run_test_eval(
        self, model_name: str, mode: Literal['validation', 'test']
    ) -> None:
        test_model(
            model_name=model_name,
            device_name=self.device_name,
            mode=mode,
            configuration=self.experiment_config,
            dataset_directory=self.dataset_directory,
            result_directory=self.results_directory,
            models_directory=self.models_directory,
        )
        evaluate_model(
            config=self.experiment_config,
            model_name=model_name,
            mode=mode,
            result_directory=self.results_directory,
            threshold=None,
        )

    def _update_from_unfinished_to_validated(
        self, model_name: str
    ) -> ExperimentSessionReport:
        return ExperimentSessionReport(
            unfinished_models=self.session_report.unfinished_models.difference(
                {model_name}
            ),
            validated_models=self.session_report.validated_models.union({model_name}),
        )

    def _validate_session_report(
        self, progress_report: ExperimentSessionReport
    ) -> ExperimentSessionReport:
        difference = self._compare_configuration_to_progress(progress_report)

        if len(difference.validated_missing_from_configuration) > 0:
            raise ValueError(
                'Found following validated models in session report '
                'without match in configuration: '
                f'{", ".join(difference.validated_missing_from_configuration)}.'
            )

        if len(difference.unfinished_missing_from_configuration) > 0:
            logger.info(
                'Following unfinished models in session report '
                'have no match in configuration. '
                f'Removing from session: '
                f'{", ".join(difference.unfinished_missing_from_configuration)}.'
            )
            progress_report = ExperimentSessionReport(
                unfinished_models=progress_report.unfinished_models.difference(
                    difference.unfinished_missing_from_configuration
                ),
                validated_models=progress_report.validated_models.copy(),
            )

        if len(difference.new_to_configuration) > 0:
            logger.info(
                'Following models have been added to the configuration since last time.'
                f'Adding to session: {", ".join(difference.new_to_configuration)}.'
            )
            progress_report = ExperimentSessionReport(
                unfinished_models=progress_report.unfinished_models.union(
                    difference.new_to_configuration
                ),
                validated_models=progress_report.validated_models.copy(),
            )

        return progress_report.copy(deep=True)

    def _compare_configuration_to_progress(
        self, progress_report: ExperimentSessionReport
    ) -> ConfigurationSessionDifference:
        models = set(self.experiment_config.models.keys())
        unfinished = progress_report.unfinished_models
        validated = progress_report.validated_models
        new_to_configuration = models.difference(unfinished, validated)
        unfinished_missing_from_configuration = unfinished.difference(models)
        validated_missing_from_configuration = validated.difference(models)

        return ConfigurationSessionDifference(
            new_to_configuration=new_to_configuration,
            unfinished_missing_from_configuration=unfinished_missing_from_configuration,
            validated_missing_from_configuration=validated_missing_from_configuration,
        )
