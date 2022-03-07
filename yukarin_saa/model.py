from typing import Dict, List, Optional

import numpy
import torch
import torch.nn.functional as F
from pytorch_trainer import report
from torch import Tensor, nn

from yukarin_saa.config import ModelConfig
from yukarin_saa.network.predictor import Predictor


class Model(nn.Module):
    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

        if self.model_config.f0_statistics_path is not None:
            statistics: Dict[str, Dict[str, float]] = numpy.load(
                self.model_config.f0_statistics_path, allow_pickle=True
            ).item()
            output_correction = numpy.array(
                [s["mean"] for s in statistics.values()], dtype=numpy.float32
            )
            self.predictor.set_output_correction(torch.from_numpy(output_correction))

    @torch.jit.export
    def f0_correction(
        self, f0: List[Tensor], voiced: List[Tensor], speaker_id: List[Tensor]
    ):
        for f0_one, voiced_one, speaker_id_one in zip(f0, voiced, speaker_id):
            corr_one = self.predictor.output_correction[speaker_id_one]
            f0_one[voiced_one] -= corr_one
        return f0

    def forward(
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
        f0 = self.f0_correction(f0=f0, voiced=voiced, speaker_id=speaker_id)

        output1, output2 = self.predictor(
            vowel_phoneme_list=vowel_phoneme,
            consonant_phoneme_list=consonant_phoneme,
            start_accent_list=start_accent,
            end_accent_list=end_accent,
            start_accent_phrase_list=start_accent_phrase,
            end_accent_phrase_list=end_accent_phrase,
            speaker_id=torch.stack(speaker_id) if speaker_id is not None else None,
        )

        mask = torch.cat(voiced)
        loss1 = F.l1_loss(torch.cat(output1)[mask], torch.cat(f0)[mask])
        loss2 = F.l1_loss(torch.cat(output2)[mask], torch.cat(f0)[mask])
        loss = loss1 + loss2

        # report
        losses = dict(loss=loss, loss1=loss1, loss2=loss2)
        if not self.training:
            weight = mask.sum()
            losses = {key: (l, weight) for key, l in losses.items()}
        report(losses, self)

        return loss
