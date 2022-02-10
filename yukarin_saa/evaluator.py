from typing import List, Optional

import numpy
import torch
from pytorch_trainer import report
from torch import Tensor, nn

from yukarin_saa.generator import Generator


class GenerateEvaluator(nn.Module):
    def __init__(
        self,
        generator: Generator,
    ):
        super().__init__()
        self.generator = generator

    def __call__(
        self,
        vowel_phoneme: List[Tensor],
        consonant_phoneme: List[Tensor],
        start_accent: List[Tensor],
        end_accent: List[Tensor],
        start_accent_phrase: List[Tensor],
        end_accent_phrase: List[Tensor],
        f0: List[Tensor],
        voiced: List[Tensor],
        speaker_id: Optional[List[Tensor]] = None,
    ):
        output = self.generator.generate(
            vowel_phoneme_list=vowel_phoneme,
            consonant_phoneme_list=consonant_phoneme,
            start_accent_list=start_accent,
            end_accent_list=end_accent,
            start_accent_phrase_list=start_accent_phrase,
            end_accent_phrase_list=end_accent_phrase,
            speaker_id=torch.stack(speaker_id) if speaker_id is not None else None,
        )

        mask = torch.cat(voiced).cpu().numpy()
        diff = numpy.abs(
            numpy.concatenate(output)[mask]
            - numpy.concatenate([t.cpu().numpy() for t in f0])[mask]
        ).mean()

        weight = mask.sum()
        scores = {"diff": (diff, weight)}

        report(scores, self)
        return scores
