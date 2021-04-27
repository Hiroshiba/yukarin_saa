from typing import Optional

import torch
from torch import Tensor, nn
from yukarin_sa.config import NetworkConfig
from yukarin_sa.network.encoder import EncoderType, create_encoder


class Predictor(nn.Module):
    def __init__(
        self,
        phoneme_size: int,
        phoneme_embedding_size: int,
        encoder_type: EncoderType,
        encoder_hidden_size: int,
        encoder_kernel_size: int,
        encoder_layer_num: int,
        speaker_size: int,
        speaker_embedding_size: int,
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

        self.encoder = create_encoder(
            type=encoder_type,
            input_size=phoneme_embedding_size + speaker_embedding_size + 4,
            hidden_size=encoder_hidden_size,
            kernel_size=encoder_kernel_size,
            layer_num=encoder_layer_num,
        )

        self.post = nn.Conv1d(self.encoder.output_hidden_size, 1, kernel_size=1)

    def forward(
        self,
        phoneme_list: Tensor,  # (batch_size, length)
        consonant_phoneme_list: Optional[Tensor],  # (batch_size, length)
        start_accent_list: Tensor,  # (batch_size, length)
        end_accent_list: Tensor,  # (batch_size, length)
        start_accent_phrase_list: Tensor,  # (batch_size, length)
        end_accent_phrase_list: Tensor,  # (batch_size, length)
        speaker_id: Optional[Tensor],  # (batch_size, )
    ):
        ph = self.phoneme_embedder(phoneme_list + 1)  # (batch_size, length, ?)
        if consonant_phoneme_list is not None:
            ph = ph + self.phoneme_embedder(consonant_phoneme_list + 1)
        ph = ph.transpose(1, 2)  # (batch_size, ?, length)

        ah = torch.stack(
            [
                start_accent_list,
                end_accent_list,
                start_accent_phrase_list,
                end_accent_phrase_list,
            ],
            dim=1,
        ).to(
            ph.dtype
        )  # (batch_size, ?, length)

        h = torch.cat((ph, ah), dim=1)  # (batch_size, ?, length)

        if self.speaker_embedder is not None and speaker_id is not None:
            speaker_id = self.speaker_embedder(speaker_id)  # (batch_size, ?)
            speaker_id = speaker_id.unsqueeze(2)  # (batch_size, ?, 1)
            speaker = speaker_id.expand(
                speaker_id.shape[0], speaker_id.shape[1], ph.shape[2]
            )  # (batch_size, ?, length)
            h = torch.cat((h, speaker), dim=1)  # (batch_size, ?, length)

        h = self.encoder(h)  # (batch_size, ?, length)
        h = self.post(h)  # (batch_size, 1, length)

        f0 = h[:, 0, :]  # (batch_size, length)
        return f0


def create_predictor(config: NetworkConfig):
    return Predictor(
        phoneme_size=config.phoneme_size,
        phoneme_embedding_size=config.phoneme_embedding_size,
        encoder_type=EncoderType(config.encoder_type),
        encoder_hidden_size=config.encoder_hidden_size,
        encoder_kernel_size=config.encoder_kernel_size,
        encoder_layer_num=config.encoder_layer_num,
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
    )
