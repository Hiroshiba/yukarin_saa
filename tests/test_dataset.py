from typing import List, Optional

import numpy
import pytest
from acoustic_feature_extractor.data.phoneme import JvsPhoneme
from yukarin_sa.dataset import f0_mean, split_mora, voiced_consonant_f0_mora


@pytest.mark.parametrize(
    "f0,rate,split_second_list,weight,expected",
    [
        (
            numpy.arange(1, 11, dtype=numpy.float32),
            2,
            [1, 2, 3, 4],
            None,
            numpy.arange(1.5, 10, step=2, dtype=numpy.float32),
        ),
        (
            numpy.array([0, 0, 0, 1, 1, 1], dtype=numpy.float32),
            2,
            [1, 2],
            None,
            numpy.array([0, 1, 1], dtype=numpy.float32),
        ),
        (
            numpy.array([1, 2, 1, 2, 1, 2], dtype=numpy.float32),
            2,
            [1, 2],
            numpy.array([0, 1, 1, 1, 1, 0], dtype=numpy.float32),
            numpy.array([2, 1.5, 1], dtype=numpy.float32),
        ),
        (
            numpy.arange(1, 11, dtype=numpy.float32),
            1.5,
            [1, 2, 3, 4],
            None,
            numpy.array([1, 2.5, 4, 5.5, 8.5], dtype=numpy.float32),
        ),
    ],
)
def test_f0_mean(
    f0: numpy.ndarray,
    rate: float,
    split_second_list: List[float],
    expected: numpy.ndarray,
    weight: numpy.ndarray,
):
    output = f0_mean(
        rate=rate,
        f0=f0,
        split_second_list=split_second_list,
        weight=weight,
    )

    numpy.testing.assert_allclose(output, expected)


@pytest.mark.parametrize(
    "phoneme_list,consonant_phoneme_list,vowel_phoneme_list,vowel_indexes",
    [
        (
            [
                JvsPhoneme("pau", 0, 1),
                JvsPhoneme("a", 1, 2),
                JvsPhoneme("k", 2, 3),
                JvsPhoneme("i", 3, 4),
                JvsPhoneme("pau", 4, 5),
            ],
            [
                None,
                None,
                JvsPhoneme("k", 2, 3),
                None,
            ],
            [
                JvsPhoneme("pau", 0, 1),
                JvsPhoneme("a", 1, 2),
                JvsPhoneme("i", 3, 4),
                JvsPhoneme("pau", 4, 5),
            ],
            [0, 1, 3, 4],
        ),
    ],
)
def test_split_mora(
    phoneme_list: List[JvsPhoneme],
    consonant_phoneme_list: List[Optional[JvsPhoneme]],
    vowel_phoneme_list: List[JvsPhoneme],
    vowel_indexes: List[int],
):
    (
        output_consonant_phoneme_list,
        output_vowel_phoneme_list,
        output_vowel_indexes,
    ) = split_mora(phoneme_list=phoneme_list)

    assert output_consonant_phoneme_list == consonant_phoneme_list
    assert output_vowel_phoneme_list == vowel_phoneme_list
    assert output_vowel_indexes == vowel_indexes


@pytest.mark.parametrize(
    "phoneme_list,f0,f0_mora",
    [
        (
            [
                JvsPhoneme("pau", 0, 1),
                JvsPhoneme("r", 1, 2),
                JvsPhoneme("a", 2, 3),
                JvsPhoneme("k", 3, 4),
                JvsPhoneme("i", 4, 5),
                JvsPhoneme("pau", 5, 6),
            ],
            numpy.array([0, 1, 2, 3, 4, 0]),
            numpy.array([0, 1.5, 4, 0]),
        ),
    ],
)
def test_voiced_consonant_f0_mora(
    phoneme_list: List[JvsPhoneme], f0: numpy.ndarray, f0_mora: numpy.ndarray
):
    output = voiced_consonant_f0_mora(phoneme_list=phoneme_list, f0=f0)
    numpy.testing.assert_array_equal(output, f0_mora)
