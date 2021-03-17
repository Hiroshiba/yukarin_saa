from pathlib import Path
from typing import List

import numpy
import pytest
from yukarin_s.dataset import f0_mean


@pytest.mark.parametrize(
    "f0,rate,split_second_list,expected",
    [
        (
            numpy.arange(1, 11, dtype=numpy.float32),
            2,
            [1, 2, 3, 4],
            numpy.arange(1.5, 10, step=2, dtype=numpy.float32),
        ),
        (
            numpy.array([0, 0, 0, 1, 1, 1], dtype=numpy.float32),
            2,
            [1, 2],
            numpy.array([0, 1, 1], dtype=numpy.float32),
        ),
        (
            numpy.arange(1, 11, dtype=numpy.float32),
            1.5,
            [1, 2, 3, 4],
            numpy.array([1, 2.5, 4, 5.5, 8.5], dtype=numpy.float32),
        ),
    ],
)
def test_f0_mean(
    f0: numpy.ndarray,
    rate: float,
    split_second_list: List[float],
    expected: numpy.ndarray,
):
    output = f0_mean(
        rate=rate,
        f0=f0,
        split_second_list=split_second_list,
    )

    numpy.testing.assert_allclose(output, expected)
