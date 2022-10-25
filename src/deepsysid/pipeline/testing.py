import dataclasses
import logging
import os
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from numpy.typing import NDArray

from ..models import utils
from ..models.base import DynamicIdentificationModel
from ..models.hybrid.bounded_residual import HybridResidualLSTMModel
from ..models.recurrent import ConstrainedRnn, LSTMInitModel, LtiRnnInit
from ..pipeline.configuration import (
    ExperimentConfiguration,
    StabilitySetting,
    initialize_model,
)
from .data_io import build_result_file_name, load_file_names, load_simulation_data
from .model_io import load_model

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ModelTestResult:
    control: List[NDArray[np.float64]]
    true_state: List[NDArray[np.float64]]
    pred_state: List[NDArray[np.float64]]
    file_names: List[str]
    metadata: List[Dict[str, NDArray[np.float64]]]


@dataclasses.dataclass
class StabilityResult:
    stability_gains: List[np.float64]
    stability_type: Literal['bibo', 'incremental']
    pred_states: List[NDArray[np.float64]]
    controls: List[NDArray[np.float64]]


def test_model(
    configuration: ExperimentConfiguration,
    model_name: str,
    device_name: str,
    mode: Literal['train', 'validation', 'test'],
    dataset_directory: str,
    result_directory: str,
    models_directory: str,
) -> None:
    model_directory = os.path.expanduser(
        os.path.normpath(os.path.join(models_directory, model_name))
    )
    model = initialize_model(configuration, model_name, device_name)
    load_model(model, model_directory, model_name)

    logger.addHandler(
        logging.FileHandler(
            filename=os.path.join(model_directory, 'testing.log'), mode='a'
        )
    )

    simulations = load_test_simulations(
        configuration=configuration,
        mode=mode,
        dataset_directory=dataset_directory,
    )

    test_result = simulate_model(
        simulations=simulations, config=configuration, model=model
    )

    save_model_tests(
        test_result=test_result,
        config=configuration,
        result_directory=result_directory,
        model_name=model_name,
        mode=mode,
    )

    if configuration.thresholds and isinstance(model, HybridResidualLSTMModel):
        for threshold in configuration.thresholds:
            test_result = simulate_model(
                simulations=simulations,
                config=configuration,
                model=model,
                threshold=threshold,
            )
            save_model_tests(
                test_result=test_result,
                config=configuration,
                result_directory=result_directory,
                model_name=model_name,
                mode=mode,
                threshold=threshold,
            )

    if configuration.test and (
        isinstance(model, LSTMInitModel)
        or isinstance(model, ConstrainedRnn)
        or isinstance(model, LtiRnnInit)
    ):
        if configuration.test.stability:
            logger.info(
                f'{model_name}: {configuration.test.stability.type} stability test.'
            )

            test_stability_and_save_reults(
                simulations=simulations,
                stability_config=configuration.test.stability,
                window_size=configuration.window_size,
                horizon_size=configuration.horizon_size,
                control_names=configuration.control_names,
                state_names=configuration.state_names,
                model=model,
                device_name=device_name,
                model_name=model_name,
                result_directory=result_directory,
                mode=mode,
            )


def save_stability_results(
    model_name: str,
    stability_results: StabilityResult,
    stability_config: StabilitySetting,
    window_size: int,
    horizon_size: int,
    control_names: List[str],
    state_names: List[str],
    result_directory: str,
    mode: Literal['train', 'validation', 'test'],
) -> None:
    # Save true and predicted time series
    result_directory = os.path.join(result_directory, model_name)
    try:
        os.mkdir(result_directory)
    except FileExistsError:
        pass

    result_file_path = os.path.join(
        result_directory,
        f'stability-{mode}'
        f'-stability_type_{stability_config.type}'
        f'-sequences_{stability_config.evaluation_sequence}'
        f'-w_{window_size}-h_{horizon_size}.hdf5',
    )

    logger = logging.getLogger(__name__)
    logger.info(f'Save stability results to {result_file_path}')

    with h5py.File(result_file_path, 'w') as f:
        write_stability_results_to_hdf5(
            f,
            control_names=control_names,
            state_names=state_names,
            stability_gains=stability_results.stability_gains,
            stability_type=stability_results.stability_type,
            pred_states=stability_results.pred_states,
            controls=stability_results.controls,
        )


def load_test_simulations(
    configuration: ExperimentConfiguration,
    mode: Literal['train', 'validation', 'test'],
    dataset_directory: str,
) -> List[Tuple[NDArray[np.float64], NDArray[np.float64], str]]:
    # Prepare test data
    dataset_directory = os.path.join(dataset_directory, 'processed', mode)

    file_names = list(
        map(
            lambda fn: os.path.basename(fn).split('.')[0],
            load_file_names(dataset_directory),
        )
    )
    controls, states = load_simulation_data(
        directory=dataset_directory,
        control_names=configuration.control_names,
        state_names=configuration.state_names,
    )

    simulations = list(zip(controls, states, file_names))

    return simulations


def simulate_model(
    simulations: List[Tuple[NDArray[np.float64], NDArray[np.float64], str]],
    config: ExperimentConfiguration,
    model: DynamicIdentificationModel,
    threshold: Optional[float] = None,
) -> ModelTestResult:
    # Execute predictions on test data
    control = []
    pred_states = []
    true_states = []
    file_names = []
    metadata = []
    for (
        initial_control,
        initial_state,
        true_control,
        true_state,
        file_name,
    ) in split_simulations(config.window_size, config.horizon_size, simulations):
        result_type = Union[
            NDArray[np.float64],
            Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]],
        ]
        if isinstance(model, HybridResidualLSTMModel):
            simulation_result: result_type = model.simulate(
                initial_control,
                initial_state,
                true_control,
                threshold=threshold if threshold is not None else np.infty,
            )
        else:
            simulation_result = model.simulate(
                initial_control, initial_state, true_control
            )

        if isinstance(simulation_result, np.ndarray):
            pred_target = simulation_result
        else:
            pred_target = simulation_result[0]
            metadata.append(simulation_result[1])

        control.append(true_control)
        pred_states.append(pred_target)
        true_states.append(true_state)
        file_names.append(file_name)

    return ModelTestResult(
        control=control,
        true_state=true_states,
        pred_state=pred_states,
        file_names=file_names,
        metadata=metadata,
    )


def save_model_tests(
    test_result: ModelTestResult,
    config: ExperimentConfiguration,
    result_directory: str,
    model_name: str,
    mode: Literal['train', 'validation', 'test'],
    threshold: Optional[float] = None,
) -> None:
    # Save true and predicted time series
    result_directory = os.path.join(result_directory, model_name)
    try:
        os.mkdir(result_directory)
    except FileExistsError:
        pass

    result_file_path = os.path.join(
        result_directory,
        build_result_file_name(
            mode, config.window_size, config.horizon_size, 'hdf5', threshold=threshold
        ),
    )
    with h5py.File(result_file_path, 'w') as f:
        write_test_results_to_hdf5(
            f,
            control_names=config.control_names,
            state_names=config.state_names,
            file_names=test_result.file_names,
            control=test_result.control,
            pred_states=test_result.pred_state,
            true_states=test_result.true_state,
            metadata=test_result.metadata,
        )


def split_simulations(
    window_size: int,
    horizon_size: int,
    simulations: List[Tuple[NDArray[np.float64], NDArray[np.float64], str]],
) -> Iterator[
    Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        str,
    ]
]:
    total_length = window_size + horizon_size
    for control, state, file_name in simulations:
        for i in range(total_length, control.shape[0], total_length):
            initial_control = control[i - total_length : i - total_length + window_size]
            initial_state = state[i - total_length : i - total_length + window_size]
            true_control = control[i - total_length + window_size : i]
            true_state = state[i - total_length + window_size : i]

            yield initial_control, initial_state, true_control, true_state, file_name


def write_stability_results_to_hdf5(
    f: h5py.File,
    control_names: List[str],
    state_names: List[str],
    stability_gains: List[np.float64],
    stability_type: Literal['bibo', 'incremental'],
    pred_states: List[NDArray[np.float64]],
    controls: List[NDArray[np.float64]],
) -> None:
    f.attrs['control_names'] = np.array([np.string_(name) for name in control_names])
    f.attrs['state_names'] = np.array([np.string_(name) for name in state_names])
    f.attrs['stability_type'] = np.string_(stability_type)

    ctr_group = f.create_group('control')
    pred_group = f.create_group('predicted')
    stab_group = f.create_group('stability_gain')

    for i, (control, pred_state, stability_gain) in enumerate(
        zip(controls, pred_states, stability_gains)
    ):
        ctr_group.create_dataset(str(i), data=control)
        pred_group.create_dataset(str(i), data=pred_state)
        stab_group.create_dataset(str(i), data=stability_gain)


def write_test_results_to_hdf5(
    f: h5py.File,
    control_names: List[str],
    state_names: List[str],
    file_names: List[str],
    control: List[NDArray[np.float64]],
    pred_states: List[NDArray[np.float64]],
    true_states: List[NDArray[np.float64]],
    metadata: List[Dict[str, NDArray[np.float64]]],
) -> None:
    f.attrs['control_names'] = np.array([np.string_(name) for name in control_names])
    f.attrs['state_names'] = np.array([np.string_(name) for name in state_names])

    f.create_dataset('file_names', data=np.array(list(map(np.string_, file_names))))

    control_grp = f.create_group('control')
    pred_grp = f.create_group('predicted')
    true_grp = f.create_group('true')

    for i, (true_control, pred_state, true_state) in enumerate(
        zip(control, pred_states, true_states)
    ):
        control_grp.create_dataset(str(i), data=true_control)
        pred_grp.create_dataset(str(i), data=pred_state)
        true_grp.create_dataset(str(i), data=true_state)

    metadata_grp = f.create_group('metadata')
    if len(metadata) > 0:
        for name in metadata[0]:
            metadata_sub_grp = metadata_grp.create_group(name)
            for i, data in enumerate(md[name] for md in metadata):
                metadata_sub_grp.create_dataset(str(i), data=data)


def test_stability_and_save_reults(
    simulations: List[Tuple[NDArray[np.float64], NDArray[np.float64], str]],
    stability_config: StabilitySetting,
    window_size: int,
    horizon_size: int,
    state_names: List[str],
    control_names: List[str],
    model: Union[LSTMInitModel, ConstrainedRnn, LtiRnnInit],
    device_name: str,
    model_name: str,
    result_directory: str,
    mode: Literal['train', 'validation', 'test'],
) -> None:
    controls = []
    pred_states = []
    stability_gains = []

    if isinstance(stability_config.evaluation_sequence, int):

        logger.info(
            f'Test stability for sequence number {stability_config.evaluation_sequence}'
        )
        dataset = [
            sequence
            for (sequence) in split_simulations(window_size, horizon_size, simulations)
        ]
        (
            initial_control,
            initial_state,
            true_control,
            _,
            _,
        ) = dataset[stability_config.evaluation_sequence]
        gamma_2, control, pred_state = optimize_input_disturbance(
            model=model,
            horizon_size=horizon_size,
            stability_config=stability_config,
            device_name=device_name,
            initial_control=initial_control,
            initial_state=initial_state,
            true_control=true_control,
        )
        stability_gains.append(gamma_2)
        controls.append(control)
        pred_states.append(pred_state)

        save_stability_results(
            model_name=model_name,
            stability_results=StabilityResult(
                stability_gains=stability_gains,
                stability_type=stability_config.type,
                pred_states=pred_states,
                controls=controls,
            ),
            state_names=state_names,
            control_names=control_names,
            window_size=window_size,
            horizon_size=horizon_size,
            stability_config=stability_config,
            result_directory=result_directory,
            mode=mode,
        )

    elif stability_config.evaluation_sequence == 'all':
        logger.info(
            f'Test stability for {stability_config.evaluation_sequence} sequences'
        )
        for (
            idx_data,
            (
                initial_control,
                initial_state,
                true_control,
                _,
                _,
            ),
        ) in enumerate(split_simulations(window_size, horizon_size, simulations)):

            logger.info(f'Sequence number: {idx_data}')

            gamma_2, control, pred_state = optimize_input_disturbance(
                model=model,
                stability_config=stability_config,
                horizon_size=horizon_size,
                device_name=device_name,
                initial_control=initial_control,
                initial_state=initial_state,
                true_control=true_control,
            )

            stability_gains.append(gamma_2)
            controls.append(control)
            pred_states.append(pred_state)

            save_stability_results(
                model_name=model_name,
                stability_results=StabilityResult(
                    stability_gains=stability_gains,
                    stability_type=stability_config.type,
                    pred_states=pred_states,
                    controls=controls,
                ),
                stability_config=stability_config,
                window_size=window_size,
                horizon_size=horizon_size,
                state_names=state_names,
                control_names=control_names,
                result_directory=result_directory,
                mode=mode,
            )


def optimize_input_disturbance(
    stability_config: StabilitySetting,
    horizon_size: int,
    model: Union[LSTMInitModel, ConstrainedRnn, LtiRnnInit],
    device_name: str,
    initial_control: NDArray[np.float64],
    initial_state: NDArray[np.float64],
    true_control: NDArray[np.float64],
) -> Tuple[np.float64, NDArray[np.float64], NDArray[np.float64]]:

    if (
        model.state_mean is None
        or model.state_std is None
        or model.control_mean is None
        or model.control_std is None
    ):
        raise ValueError('Mean and standard deviation is not initialized in the model')

    model.predictor.train()

    # normalize data
    u_init_norm_numpy = utils.normalize(
        initial_control, model.control_mean, model.control_std
    )
    u_norm_numpy = utils.normalize(true_control, model.control_mean, model.control_std)
    y_init_norm_numpy = utils.normalize(
        initial_state, model.state_mean, model.state_std
    )

    # convert to tensors
    u_init_norm = (
        torch.from_numpy(np.hstack((u_init_norm_numpy[1:], y_init_norm_numpy[:-1])))
        .unsqueeze(0)
        .float()
        .to(device_name)
    )
    u_norm = torch.from_numpy(u_norm_numpy).unsqueeze(0).float().to(device_name)

    # disturb input
    delta = torch.normal(
        stability_config.initial_mean_delta,
        stability_config.initial_std_delta,
        size=(horizon_size, model.control_dim),
        requires_grad=True,
        device=device_name,
    )

    # optimizer
    opt = torch.optim.Adam(  # type: ignore
        [delta], lr=stability_config.optimization_lr, maximize=True
    )

    for step_idx in range(stability_config.optimization_steps):

        # calculate stability gain
        def sequence_norm(x):
            norm = torch.tensor(0).float().to(device_name)
            for x_k in x:
                norm += (torch.linalg.norm(x_k) ** 2).float()
            return norm

        u_a = u_norm + delta
        if stability_config.type == 'incremental':
            u_b = u_norm.clone()
            # model prediction
            _, hx = model.initializer(u_init_norm)
            # TODO set initial state to zero should be good to find unstable sequences
            hx = (
                torch.zeros_like(hx[0]).to(device_name),
                torch.zeros_like(hx[1]).to(device_name),
            )
            y_hat_a, _ = model.predictor(u_a, hx=hx)
            y_hat_b, _ = model.predictor(u_b, hx=hx)
            y_hat_a = y_hat_a.squeeze()
            y_hat_b = y_hat_b.squeeze()

            # use log to avoid zero in the denominator (goes to -inf)
            # since we maximize this results in a punishment
            regularization = stability_config.regularization_scale * torch.log(
                sequence_norm(u_a - u_b)
            )
            gamma_2 = sequence_norm(y_hat_a - y_hat_b) / sequence_norm(u_a - u_b)
            L = gamma_2 + regularization
            L.backward()
            torch.nn.utils.clip_grad_norm_(delta, stability_config.clip_gradient_norm)
            opt.step()

            control = utils.denormalize(
                (u_a - u_b).cpu().detach().numpy().squeeze(),
                model.control_mean,
                model.control_std,
            )
            pred_state = utils.denormalize(
                (y_hat_a - y_hat_b).cpu().detach().numpy(),
                model.state_mean,
                model.state_std,
            )

        elif stability_config.type == 'bibo':
            # model prediction
            _, hx = model.initializer(u_init_norm)
            # TODO set initial state to zero should be good to find unstable sequences
            hx = (
                torch.zeros_like(hx[0]).to(device_name),
                torch.zeros_like(hx[1]).to(device_name),
            )
            y_hat_a, _ = model.predictor(u_a, hx=hx)
            y_hat_a = y_hat_a.squeeze()

            # use log to avoid zero in the denominator (goes to -inf)
            # since we maximize this results in a punishment
            regularization = stability_config.regularization_scale * torch.log(
                sequence_norm(u_a)
            )
            gamma_2 = sequence_norm(y_hat_a) / sequence_norm(u_a)
            L = gamma_2 + regularization
            L.backward()
            torch.nn.utils.clip_grad_norm_(delta, stability_config.clip_gradient_norm)
            # torch.nn.utils.clip_grad_norm_(delta, 100)
            opt.step()

            control = utils.denormalize(
                u_a.cpu().detach().numpy().squeeze(),
                model.control_mean,
                model.control_std,
            )
            pred_state = utils.denormalize(
                y_hat_a.cpu().detach().numpy(), model.state_mean, model.state_std
            )

        gamma_2 = gamma_2.cpu().detach().numpy()
        if step_idx % 100 == 0:
            logger.info(
                f'step: {step_idx} \t '
                f'gamma_2: {gamma_2:.3f} \t '
                f'gradient norm: {torch.norm(delta.grad):.3f} '
                f'\t -log(norm(denominator)): {regularization:.3f}'
            )

    return gamma_2, control, pred_state
