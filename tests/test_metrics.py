import numpy as np
import pytest

from deepsysid.utils import index_of_agreement, score_on_sequence


def test_index_of_agreement_empty_arrays():
    a1 = np.array([])
    a2 = np.array([])

    with pytest.raises(ValueError):
        index_of_agreement(a1, a2, j=1)


def test_index_of_agreement_mismatch_arrays():
    a1 = np.array([[1, 2], [3, 4]])
    a2 = np.array([[1, 2], [3, 4], [5, 6]])

    with pytest.raises(ValueError):
        index_of_agreement(a1, a2, j=1)


def test_index_of_agreement_incorrect_exponent():
    a1 = np.array([[1, 2], [3, 4]])
    a2 = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError):
        index_of_agreement(a1, a2, j=0)


def test_index_of_agreement_perfect_agreement():
    a1 = np.array([[1, 2], [3, 4]])
    a2 = np.array([[1, 2], [3, 4]])

    assert index_of_agreement(a1, a2, j=1) == pytest.approx(1.0)
