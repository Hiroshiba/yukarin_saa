from typing import Optional

import numpy
import torch
from pytorch_trainer import report
from torch import Tensor, nn

from yukarin_sa.generator import Generator


class GenerateEvaluator(nn.Module):
    def __init__(
        self,
        generator: Generator,
    ):
        super().__init__()
        self.generator = generator

    def __call__(
        self,
        vowel_phoneme_list: Tensor,
        consonant_phoneme_list: Tensor,
        start_accent_list: Tensor,
        end_accent_list: Tensor,
        start_accent_phrase_list: Tensor,
        end_accent_phrase_list: Tensor,
        f0: Tensor,
        voiced: Tensor,
        padded: Tensor,
        speaker_id: Optional[Tensor] = None,
    ):
        batch_size = vowel_phoneme_list.shape[0]
        numpy_mask = torch.logical_and(voiced, ~padded).cpu().numpy()

        out_f0 = self.generator.generate(
            vowel_phoneme_list=vowel_phoneme_list,
            consonant_phoneme_list=consonant_phoneme_list,
            start_accent_list=start_accent_list,
            end_accent_list=end_accent_list,
            start_accent_phrase_list=start_accent_phrase_list,
            end_accent_phrase_list=end_accent_phrase_list,
            speaker_id=speaker_id,
        )
        out_f0 = out_f0[numpy_mask]

        in_f0 = f0.cpu().numpy()[numpy_mask]

        diff = numpy.abs(out_f0 - in_f0).mean()

        weight = (numpy_mask).mean() * batch_size
        scores = {"diff": (diff, weight)}

        report(scores, self)
        return scores
