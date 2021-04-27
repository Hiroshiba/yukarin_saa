from pathlib import Path
from typing import Optional, Union

import numpy
import torch

from yukarin_sa.config import Config
from yukarin_sa.network.predictor import Predictor, create_predictor


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
            state_dict = torch.load(predictor, map_location=self.device)
            predictor = create_predictor(config.network)
            predictor.load_state_dict(state_dict)
        self.predictor = predictor.eval().to(self.device)

    def generate(
        self,
        phoneme_list: Union[numpy.ndarray, torch.Tensor],
        consonant_phoneme_list: Optional[Union[numpy.ndarray, torch.Tensor]],
        start_accent_list: Union[numpy.ndarray, torch.Tensor],
        end_accent_list: Union[numpy.ndarray, torch.Tensor],
        start_accent_phrase_list: Union[numpy.ndarray, torch.Tensor],
        end_accent_phrase_list: Union[numpy.ndarray, torch.Tensor],
        speaker_id: Optional[Union[int, numpy.ndarray, torch.Tensor]],
    ):
        if isinstance(phoneme_list, numpy.ndarray):
            phoneme_list = torch.from_numpy(phoneme_list)
        phoneme_list = phoneme_list.unsqueeze(0).to(self.device)

        if consonant_phoneme_list is not None:
            if isinstance(consonant_phoneme_list, numpy.ndarray):
                consonant_phoneme_list = torch.from_numpy(consonant_phoneme_list)
            consonant_phoneme_list = consonant_phoneme_list.unsqueeze(0).to(self.device)

        if isinstance(start_accent_list, numpy.ndarray):
            start_accent_list = torch.from_numpy(start_accent_list)
        start_accent_list = start_accent_list.unsqueeze(0).to(self.device)

        if isinstance(end_accent_list, numpy.ndarray):
            end_accent_list = torch.from_numpy(end_accent_list)
        end_accent_list = end_accent_list.unsqueeze(0).to(self.device)

        if isinstance(start_accent_phrase_list, numpy.ndarray):
            start_accent_phrase_list = torch.from_numpy(start_accent_phrase_list)
        start_accent_phrase_list = start_accent_phrase_list.unsqueeze(0).to(self.device)

        if isinstance(end_accent_phrase_list, numpy.ndarray):
            end_accent_phrase_list = torch.from_numpy(end_accent_phrase_list)
        end_accent_phrase_list = end_accent_phrase_list.unsqueeze(0).to(self.device)

        if speaker_id is not None:
            if isinstance(speaker_id, int):
                speaker_id = numpy.array(speaker_id)
            if isinstance(speaker_id, numpy.ndarray):
                speaker_id = torch.from_numpy(speaker_id)
            speaker_id = speaker_id.reshape((1,)).to(torch.int64).to(self.device)

        with torch.no_grad():
            f0 = self.predictor(
                phoneme_list=phoneme_list,
                consonant_phoneme_list=consonant_phoneme_list,
                start_accent_list=start_accent_list,
                end_accent_list=end_accent_list,
                start_accent_phrase_list=start_accent_phrase_list,
                end_accent_phrase_list=end_accent_phrase_list,
                speaker_id=speaker_id,
            )

        f0 = f0[0].cpu().numpy()
        return f0
