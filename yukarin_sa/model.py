from typing import Dict, Optional

import numpy
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

    def __call__(
        self,
        phoneme_list: Tensor,
        phoneme_length: Tensor,
        padded: Tensor,
        start_accent: Tensor,
        end_accent: Tensor,
        f0: Tensor,
        speaker_id: Optional[Tensor] = None,
    ):
        batch_size = len(phoneme_list)

        output_phoneme_length, output_f0 = self.predictor(
            phoneme_list=phoneme_list,
            start_accent=start_accent,
            end_accent=end_accent,
            speaker_id=speaker_id,
        )

        values: Dict[str, Tensor] = {}

        pl_loss = F.l1_loss(
            output_phoneme_length[~padded], phoneme_length[~padded], reduction="none"
        )
        f0_loss = F.l1_loss(output_f0[~padded], f0[~padded], reduction="none")

        if self.model_config.eliminate_pause:
            pl_loss = pl_loss[phoneme_list != 0]
            f0_loss = f0_loss[phoneme_list != 0]

        pl_loss = pl_loss.mean() * self.model_config.phoneme_length_loss_weight
        f0_loss = f0_loss.mean() * self.model_config.f0_loss_weight

        loss = pl_loss + f0_loss

        values = dict(
            loss=loss,
            pl_loss=pl_loss,
            f0_loss=f0_loss,
        )

        # report
        if not self.training:
            weight = (~padded).to(torch.float32).mean() * batch_size
            values = {key: (l, weight) for key, l in values.items()}  # add weight
        report(values, self)

        return loss
