from pathlib import Path
from typing import Any, List, Optional, Union

import numpy
import torch
from torch import Tensor

from yukarin_saa.config import Config
from yukarin_saa.network.predictor import Predictor, create_predictor


def to_tensor(array: Union[Tensor, numpy.ndarray, Any]):
    if not isinstance(array, (Tensor, numpy.ndarray)):
        array = numpy.asarray(array)
    if isinstance(array, numpy.ndarray):
        return torch.from_numpy(array)
    else:
        return array


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
        vowel_phoneme_list: List[Union[numpy.ndarray, torch.Tensor]],
        consonant_phoneme_list: List[Union[numpy.ndarray, torch.Tensor]],
        start_accent_list: List[Union[numpy.ndarray, torch.Tensor]],
        end_accent_list: List[Union[numpy.ndarray, torch.Tensor]],
        start_accent_phrase_list: List[Union[numpy.ndarray, torch.Tensor]],
        end_accent_phrase_list: List[Union[numpy.ndarray, torch.Tensor]],
        speaker_id: Optional[Union[numpy.ndarray, torch.Tensor]],
    ):
        vowel_phoneme_list = [to_tensor(t).to(self.device) for t in vowel_phoneme_list]
        consonant_phoneme_list = [
            to_tensor(t).to(self.device) for t in consonant_phoneme_list
        ]
        start_accent_list = [to_tensor(t).to(self.device) for t in start_accent_list]
        end_accent_list = [to_tensor(t).to(self.device) for t in end_accent_list]
        start_accent_phrase_list = [
            to_tensor(t).to(self.device) for t in start_accent_phrase_list
        ]
        end_accent_phrase_list = [
            to_tensor(t).to(self.device) for t in end_accent_phrase_list
        ]
        if speaker_id is not None:
            speaker_id = to_tensor(speaker_id).to(self.device)

        with torch.no_grad():
            output_list = self.predictor.inference(
                vowel_phoneme_list=vowel_phoneme_list,
                consonant_phoneme_list=consonant_phoneme_list,
                start_accent_list=start_accent_list,
                end_accent_list=end_accent_list,
                start_accent_phrase_list=start_accent_phrase_list,
                end_accent_phrase_list=end_accent_phrase_list,
                speaker_id=speaker_id,
            )
        return [output.cpu().numpy() for output in output_list]
