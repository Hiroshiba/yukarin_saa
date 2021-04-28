from typing import Optional

import torch
import torch.nn.functional as F
from pytorch_trainer import report
from torch import Tensor, nn

from yukarin_sa.config import ModelConfig
from yukarin_sa.network.predictor import Predictor


class Model(nn.Module):
    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def forward(
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
        batch_size = len(vowel_phoneme_list)

        output_f0 = self.predictor(
            vowel_phoneme_list=vowel_phoneme_list,
            consonant_phoneme_list=consonant_phoneme_list,
            start_accent_list=start_accent_list,
            end_accent_list=end_accent_list,
            start_accent_phrase_list=start_accent_phrase_list,
            end_accent_phrase_list=end_accent_phrase_list,
            speaker_id=speaker_id,
        )

        mask = torch.logical_and(voiced, ~padded)
        f0_loss = F.l1_loss(output_f0[mask], f0[mask], reduction="none")
        f0_loss = f0_loss.mean() * self.model_config.f0_loss_weight
        loss = f0_loss

        values = dict(loss=loss, f0_loss=f0_loss)

        # report
        if not self.training:
            weight = (~padded).to(torch.float32).mean() * batch_size
            values = {key: (l, weight) for key, l in values.items()}  # add weight
        report(values, self)

        return loss
