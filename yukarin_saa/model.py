from typing import List, Optional

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
        batch_size = len(vowel_phoneme)

        output1, output2 = self.predictor(
            vowel_phoneme_list=vowel_phoneme,
            consonant_phoneme_list=consonant_phoneme,
            start_accent_list=start_accent,
            end_accent_list=end_accent,
            start_accent_phrase_list=start_accent_phrase,
            end_accent_phrase_list=end_accent_phrase,
            speaker_id=speaker_id,
        )

        mask = torch.cat(voiced)
        loss1 = F.l1_loss(torch.cat(output1)[mask], torch.cat(f0)[mask])
        loss2 = F.l1_loss(torch.cat(output2)[mask], torch.cat(f0)[mask])
        loss = loss1 + loss2

        # report
        losses = dict(loss=loss, loss1=loss1, loss2=loss2)
        if not self.training:
            losses = {key: (l, batch_size) for key, l in losses.items()}
        report(losses, self)

        return loss
