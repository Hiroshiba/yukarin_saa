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
        ar_encoder_type: Optional[EncoderType],
        ar_encoder_hidden_size: int,
        ar_encoder_kernel_size: int,
        ar_encoder_layer_num: int,
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

        self.ar_encoder = (
            create_encoder(
                type=ar_encoder_type,
                input_size=self.encoder.output_hidden_size + 1,
                hidden_size=ar_encoder_hidden_size,
                kernel_size=ar_encoder_kernel_size,
                layer_num=ar_encoder_layer_num,
            )
            if ar_encoder_type is not None
            else None
        )

        input_size = (
            self.encoder.output_hidden_size
            if self.ar_encoder is None
            else self.ar_encoder.output_hidden_size
        )
        self.post = nn.Conv1d(input_size, 1, kernel_size=1)

    def forward_encoder(
        self,
        vowel_phoneme_list: Tensor,  # (batch_size, length)
        consonant_phoneme_list: Tensor,  # (batch_size, length)
        start_accent_list: Tensor,  # (batch_size, length)
        end_accent_list: Tensor,  # (batch_size, length)
        start_accent_phrase_list: Tensor,  # (batch_size, length)
        end_accent_phrase_list: Tensor,  # (batch_size, length)
        speaker_id: Optional[Tensor],  # (batch_size, )
    ):
        ph = self.phoneme_embedder(vowel_phoneme_list + 1) + self.phoneme_embedder(
            consonant_phoneme_list + 1
        )  # (batch_size, length, ?)
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

        return self.encoder(h)  # (batch_size, ?, length)

    def forward(
        self,
        vowel_phoneme_list: Tensor,  # (batch_size, length)
        consonant_phoneme_list: Tensor,  # (batch_size, length)
        start_accent_list: Tensor,  # (batch_size, length)
        end_accent_list: Tensor,  # (batch_size, length)
        start_accent_phrase_list: Tensor,  # (batch_size, length)
        end_accent_phrase_list: Tensor,  # (batch_size, length)
        f0: Tensor,  # (batch_size, length)
        speaker_id: Optional[Tensor],  # (batch_size, )
    ):
        h = self.forward_encoder(
            vowel_phoneme_list=vowel_phoneme_list,
            consonant_phoneme_list=consonant_phoneme_list,
            start_accent_list=start_accent_list,
            end_accent_list=end_accent_list,
            start_accent_phrase_list=start_accent_phrase_list,
            end_accent_phrase_list=end_accent_phrase_list,
            speaker_id=speaker_id,
        )  # (batch_size, ?, length)

        if self.ar_encoder is not None:
            h = torch.cat((h, f0.unsqueeze(1)), dim=1)  # (batch_size, ?, length)
            h = self.ar_encoder(h)  # (batch_size, ?, length)

        h = self.post(h)  # (batch_size, 1, length)
        return h[:, 0, :]  # (batch_size, length)

    def inference(
        self,
        vowel_phoneme_list: Tensor,  # (batch_size, length)
        consonant_phoneme_list: Tensor,  # (batch_size, length)
        start_accent_list: Tensor,  # (batch_size, length)
        end_accent_list: Tensor,  # (batch_size, length)
        start_accent_phrase_list: Tensor,  # (batch_size, length)
        end_accent_phrase_list: Tensor,  # (batch_size, length)
        speaker_id: Optional[Tensor],  # (batch_size, )
    ):
        batch_size = vowel_phoneme_list.shape[0]
        length = vowel_phoneme_list.shape[1]

        h = self.forward_encoder(
            vowel_phoneme_list=vowel_phoneme_list,
            consonant_phoneme_list=consonant_phoneme_list,
            start_accent_list=start_accent_list,
            end_accent_list=end_accent_list,
            start_accent_phrase_list=start_accent_phrase_list,
            end_accent_phrase_list=end_accent_phrase_list,
            speaker_id=speaker_id,
        )  # (batch_size, ?, length)

        if self.ar_encoder is not None:
            f0 = torch.zeros(
                batch_size, length, dtype=h.dtype, device=h.device
            )  # (batch_size, length)

            f0_one = torch.zeros(
                batch_size, 1, 1, dtype=h.dtype, device=h.device
            )  # (batch_size, 1, 1)
            hidden = None
            for i in range(length):
                h_one = h[:, :, i : i + 1]  # (batch_size, ?, 1)
                h_one = torch.cat((h_one, f0_one), dim=1)  # (batch_size, ?, 1)
                h_one, hidden = self.ar_encoder(
                    h_one, hidden=hidden, return_hidden=True
                )  # (batch_size, ?, 1)
                f0_one = self.post(h_one)  # (batch_size, 1, 1)

                f0[:, i] = f0_one[:, 0, 0]  # (batch_size, length)

        else:
            h = self.post(h)  # (batch_size, 1, length)
            f0 = h[:, 0, :]  # (batch_size, length)

        return f0  # (batch_size, length)


def create_predictor(config: NetworkConfig):
    return Predictor(
        phoneme_size=config.phoneme_size,
        phoneme_embedding_size=config.phoneme_embedding_size,
        encoder_type=EncoderType(config.encoder_type),
        encoder_hidden_size=config.encoder_hidden_size,
        encoder_kernel_size=config.encoder_kernel_size,
        encoder_layer_num=config.encoder_layer_num,
        ar_encoder_type=(
            EncoderType(config.ar_encoder_type)
            if config.ar_encoder_type is not None
            else None
        ),
        ar_encoder_hidden_size=config.ar_encoder_hidden_size,
        ar_encoder_kernel_size=config.ar_encoder_kernel_size,
        ar_encoder_layer_num=config.ar_encoder_layer_num,
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
    )
