from typing import Optional, Sequence

import numpy
import torch
import torch.nn.functional as F
from pytorch_trainer import report
from torch import Tensor, nn

from yukarin_s.config import ModelConfig
from yukarin_s.network.predictor import Predictor


class Model(nn.Module):
    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def __call__(
        self,
        phoneme_list: Tensor,
        phoneme_length: Tensor,
        speaker_id: Optional[Tensor] = None,
    ):
        batch_size = len(phoneme_list)

        output = self.predictor(
            phoneme_list=phoneme_list,
            speaker_id=speaker_id,
        )

        loss = F.l1_loss(output, phoneme_length, reduction="none")
        if self.model_config.eliminate_pause:
            loss = loss[phoneme_list != 0]
        loss = loss.mean()

        # report
        values = dict(loss=loss)
        if not self.training:
            weight = batch_size
            values = {key: (l, weight) for key, l in values.items()}  # add weight
        report(values, self)

        return loss
