from pathlib import Path
from typing import Dict, List, Optional, OrderedDict

import numpy
import torch
from espnet_pytorch_library.conformer.encoder import Encoder
from espnet_pytorch_library.nets_utils import make_non_pad_mask
from espnet_pytorch_library.tacotron2.decoder import Postnet
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from yukarin_saa.config import NetworkConfig


class Predictor(nn.Module):
    def __init__(
        self,
        phoneme_size: int,
        phoneme_embedding_size: int,
        speaker_size: int,
        speaker_embedding_size: int,
        hidden_size: int,
        block_num: int,
        post_layer_num: int,
    ):
        super().__init__()

        self.phoneme_embedder = nn.Embedding(
            num_embeddings=phoneme_size + 1,  # empty consonant
            embedding_dim=phoneme_embedding_size,
            padding_idx=0,
        )
        self.speaker_embedder = (
            nn.Embedding(
                num_embeddings=speaker_size,
                embedding_dim=speaker_embedding_size,
            )
            if speaker_size > 0
            else None
        )

        input_size = (
            phoneme_embedding_size
            + 4
            + (speaker_embedding_size if self.speaker_embedder else 0)
        )
        self.pre = torch.nn.Linear(input_size, hidden_size)

        self.encoder = Encoder(
            idim=None,
            attention_dim=hidden_size,
            attention_heads=2,
            linear_units=hidden_size * 4,
            num_blocks=block_num,
            input_layer=None,
            dropout_rate=0.2,
            positional_dropout_rate=0.2,
            attention_dropout_rate=0.2,
            normalize_before=True,
            positionwise_layer_type="conv1d",
            positionwise_conv_kernel_size=3,
            macaron_style=True,
            pos_enc_layer_type="rel_pos",
            selfattention_layer_type="rel_selfattn",
            activation_type="swish",
            use_cnn_module=True,
            cnn_module_kernel=31,
        )

        self.post = torch.nn.Linear(hidden_size, 1)

        if post_layer_num > 0:
            self.postnet = Postnet(
                idim=None,
                odim=1,
                n_layers=post_layer_num,
                n_chans=hidden_size,
                n_filts=5,
                use_batch_norm=True,
                dropout_rate=0.5,
            )
        else:
            self.postnet = None

        self.output_correction: Tensor
        self.register_buffer("output_correction", torch.zeros(speaker_size))

    def set_output_correction(self, output_correction: Tensor):
        self.output_correction[:] = output_correction

    def _mask(self, length: Tensor):
        x_masks = make_non_pad_mask(length).to(length.device)
        return x_masks.unsqueeze(-2)

    def forward(
        self,
        vowel_phoneme_list: List[Tensor],  # [(length, )]
        consonant_phoneme_list: List[Tensor],  # [(length, )]
        start_accent_list: List[Tensor],  # [(length, )]
        end_accent_list: List[Tensor],  # [(length, )]
        start_accent_phrase_list: List[Tensor],  # [(length, )]
        end_accent_phrase_list: List[Tensor],  # [(length, )]
        speaker_id: Optional[Tensor],  # (batch_size, )
    ):
        length_list = [t.shape[0] for t in vowel_phoneme_list]

        length = torch.tensor(length_list, device=vowel_phoneme_list[0].device)
        vowel_phoneme = pad_sequence(vowel_phoneme_list, batch_first=True)
        consonant_phoneme = pad_sequence(consonant_phoneme_list, batch_first=True)
        start_accent = pad_sequence(start_accent_list, batch_first=True)
        end_accent = pad_sequence(end_accent_list, batch_first=True)
        start_accent_phrase = pad_sequence(start_accent_phrase_list, batch_first=True)
        end_accent_phrase = pad_sequence(end_accent_phrase_list, batch_first=True)

        ph = self.phoneme_embedder(vowel_phoneme + 1) + self.phoneme_embedder(
            consonant_phoneme + 1
        )  # (batch_size, length, ?)

        ah = torch.stack(
            [start_accent, end_accent, start_accent_phrase, end_accent_phrase],
            dim=2,
        ).to(
            ph.dtype
        )  # (batch_size, length, ?)

        h = torch.cat((ph, ah), dim=2)  # (batch_size, length, ?)

        if self.speaker_embedder is not None and speaker_id is not None:
            s = self.speaker_embedder(speaker_id)  # (batch_size, ?)
            s = s.unsqueeze(1)  # (batch_size, 1, ?)
            s = s.expand(s.shape[0], ph.shape[1], s.shape[2])  # (batch_size, length, ?)
            h = torch.cat((h, s), dim=2)  # (batch_size, length, ?)

        h = self.pre(h)

        mask = self._mask(length)
        h, _ = self.encoder(h, mask)

        output1 = self.post(h)
        if self.postnet is not None:
            output2 = output1 + self.postnet(output1.transpose(1, 2)).transpose(1, 2)
        else:
            output2 = output1

        return (
            [output1[i, :l, 0] for i, l in enumerate(length_list)],
            [output2[i, :l, 0] for i, l in enumerate(length_list)],
        )

    def inference(
        self,
        vowel_phoneme_list: List[Tensor],
        consonant_phoneme_list: List[Tensor],
        start_accent_list: List[Tensor],
        end_accent_list: List[Tensor],
        start_accent_phrase_list: List[Tensor],
        end_accent_phrase_list: List[Tensor],
        speaker_id: Optional[Tensor],
    ):
        _, h = self(
            vowel_phoneme_list=vowel_phoneme_list,
            consonant_phoneme_list=consonant_phoneme_list,
            start_accent_list=start_accent_list,
            end_accent_list=end_accent_list,
            start_accent_phrase_list=start_accent_phrase_list,
            end_accent_phrase_list=end_accent_phrase_list,
            speaker_id=speaker_id,
        )
        if speaker_id is not None:
            for h_one, s_one in zip(h, speaker_id):
                h_one += self.output_correction[s_one]
        return h


def create_predictor(config: NetworkConfig):
    return Predictor(
        phoneme_size=config.phoneme_size,
        phoneme_embedding_size=config.phoneme_embedding_size,
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
        hidden_size=config.hidden_size,
        block_num=config.block_num,
        post_layer_num=config.post_layer_num,
    )
