from typing import List, Optional

import torch
from torch import Tensor, nn
from yukarin_s.config import NetworkConfig
from yukarin_s.network.encoder import EncoderType, create_encoder


class Predictor(nn.Module):
    def __init__(
        self,
        phoneme_size: int,
        phoneme_embedding_size: int,
        speaker_size: int,
        speaker_embedding_size: int,
        encoder_type: EncoderType,
        encoder_hidden_size: int,
        encoder_kernel_size: int,
        encoder_layer_num: int,
        estimate_f0_flag: bool,
    ):
        super().__init__()

        self.with_speaker = speaker_size > 0
        self.phoneme_size = phoneme_size
        self.estimate_f0_flag = estimate_f0_flag

        self.phoneme_embedder = nn.Embedding(
            num_embeddings=phoneme_size,
            embedding_dim=phoneme_embedding_size,
        )
        self.speaker_embedder = (
            nn.Embedding(
                num_embeddings=speaker_size,
                embedding_dim=speaker_embedding_size,
            )
            if self.with_speaker
            else None
        )

        input_size = phoneme_embedding_size + speaker_embedding_size

        self.encoder = create_encoder(
            type=encoder_type,
            input_size=input_size,
            hidden_size=encoder_hidden_size,
            kernel_size=encoder_kernel_size,
            layer_num=encoder_layer_num,
        )

        output_size = 1 if not estimate_f0_flag else 2
        self.post = nn.Conv1d(
            self.encoder.output_hidden_size, output_size, kernel_size=1
        )

    def forward(
        self,
        phoneme_list: Tensor,  # (batch_size, length)
        speaker_id: Optional[Tensor],  # (batch_size, )
    ):
        h = self.phoneme_embedder(phoneme_list)  # (batch_size, length, ?)
        h = h.transpose(1, 2)  # (batch_size, ?, length)

        if self.with_speaker:
            speaker_id = self.speaker_embedder(speaker_id)  # (batch_size, ?)
            speaker_id = speaker_id.unsqueeze(2)  # (batch_size, ?, 1)
            speaker = speaker_id.expand(
                speaker_id.shape[0], speaker_id.shape[1], h.shape[2]
            )  # (batch_size, ?, length)
            h = torch.cat((h, speaker), dim=1)  # (batch_size, ?, length)

        h = self.encoder(h)  # (batch_size, ?, length)
        h = self.post(h)  # (batch_size, ?, length)

        phoneme_length = h[:, 0, :]  # (batch_size, length)

        f0: Optional[Tensor] = None
        if self.estimate_f0_flag:
            f0 = h[:, 1, :]  # (batch_size, length)

        return phoneme_length, f0


def create_predictor(config: NetworkConfig):
    return Predictor(
        phoneme_size=config.phoneme_size,
        phoneme_embedding_size=config.phoneme_embedding_size,
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
        encoder_type=EncoderType(config.encoder_type),
        encoder_hidden_size=config.encoder_hidden_size,
        encoder_kernel_size=config.encoder_kernel_size,
        encoder_layer_num=config.encoder_layer_num,
        estimate_f0_flag=config.estimate_f0_flag,
    )
