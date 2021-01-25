from pathlib import Path
from typing import Optional, Union

import numpy
import torch

from yukarin_s.config import Config
from yukarin_s.network.predictor import Predictor, create_predictor


class Generator(object):
    def __init__(
        self,
        config: Config,
        predictor: Union[Predictor, Path],
        use_gpu: bool = True,
    ):
        self.config = config
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")

        if isinstance(predictor, Path):
            state_dict = torch.load(predictor)
            predictor = create_predictor(config.network)
            predictor.load_state_dict(state_dict)
        self.predictor = predictor.eval().to(self.device)

    def generate(
        self,
        phoneme_list: Union[numpy.ndarray, torch.Tensor],
        speaker_id: Optional[Union[int, numpy.ndarray, torch.Tensor]],
    ):
        if isinstance(phoneme_list, numpy.ndarray):
            phoneme_list = torch.from_numpy(phoneme_list)
        phoneme_list = phoneme_list.to(self.device)

        if speaker_id is not None:
            if isinstance(speaker_id, int):
                speaker_id = numpy.array(speaker_id)
            if isinstance(speaker_id, numpy.ndarray):
                speaker_id = torch.from_numpy(speaker_id)
            speaker_id = speaker_id.reshape((1,)).to(torch.int64).to(self.device)

        output = self.predictor(
            phoneme_list=phoneme_list.unsqueeze(0),
            speaker_id=speaker_id.unsqueeze(0),
        )
        return output.cpu().numpy()
