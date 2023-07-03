from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ..utils import TensorType
from .physical import PhysicalComponent
from .semiphysical import SemiphysicalComponent


class ArrayVariableMapper:
    def __init__(
        self, provided_variables: List[str], expected_variables: Optional[List[str]]
    ) -> None:
        if expected_variables is not None and any(
            sn not in provided_variables for sn in expected_variables
        ):
            raise ValueError(
                f'Expected variables {expected_variables} '
                f'but receives variables {provided_variables}.'
            )

        if expected_variables is None:
            self.mask: NDArray[np.int32] = np.array(
                list(range(len(provided_variables))), dtype=np.int32
            )
        else:
            self.mask = np.array(
                list(provided_variables.index(var) for var in expected_variables),
                dtype=np.int32,
            )

    def get_expected(self, provided: TensorType) -> TensorType:
        return provided[..., self.mask]

    def set_provided(self, provided_target: TensorType, expected: TensorType) -> None:
        provided_target[..., self.mask] = expected


class ComponentMapper:
    def __init__(
        self,
        control_variables: List[str],
        state_variables: List[str],
        component: Union[PhysicalComponent, SemiphysicalComponent],
    ) -> None:
        self.control_mapper = ArrayVariableMapper(
            provided_variables=control_variables, expected_variables=component.CONTROLS
        )
        self.state_mapper = ArrayVariableMapper(
            provided_variables=state_variables, expected_variables=component.STATES
        )

    def get_expected_control(self, provided: TensorType) -> TensorType:
        return self.control_mapper.get_expected(provided)

    def get_expected_state(self, provided: TensorType) -> TensorType:
        return self.state_mapper.get_expected(provided)

    def set_provided_control(
        self, provided_target: TensorType, expected: TensorType
    ) -> None:
        self.control_mapper.set_provided(provided_target, expected)

    def set_provided_state(
        self, provided_target: TensorType, expected: TensorType
    ) -> None:
        self.state_mapper.set_provided(provided_target, expected)
