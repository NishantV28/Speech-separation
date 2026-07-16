"""STFT encoder / iSTFT decoder. Shared by every pipeline.

Frozen convention (must not differ across pipelines, or no comparison is valid):

    wav  (B, T)              --encode-->  X (B, 2, F, T')     RI stacked
    X    (B, 2, F, T')       --decode-->  wav (B, T)

n_fft=256 / hop=128 at 16 kHz gives F=129, T'=501 for a 4 s clip. TF-GridNet
uses 512/128 at 16 kHz; 256 halves the sub-band cost, which is what makes a
5M-param model tractable on a T4. Tunable -- but tune it in the config, once,
for all seven pipelines at the same time.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class STFT(nn.Module):
    def __init__(self, n_fft: int = 256, hop: int = 128, win_length: int | None = None):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.win_length = win_length or n_fft
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)

    @property
    def n_freqs(self) -> int:
        return self.n_fft // 2 + 1

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, T) -> (B, 2, F, T')"""
        X = torch.stft(
            wav,
            self.n_fft,
            self.hop,
            self.win_length,
            self.window,
            center=True,
            return_complex=True,
        )
        return torch.stack([X.real, X.imag], dim=1)

    def decode(self, X: torch.Tensor, length: int) -> torch.Tensor:
        """(B, 2, F, T') -> (B, T)"""
        Xc = torch.complex(X[:, 0], X[:, 1])
        return torch.istft(
            Xc,
            self.n_fft,
            self.hop,
            self.win_length,
            self.window,
            center=True,
            length=length,
        )

    def apply_mask(self, X: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Complex ratio masking.

        X     (B, 2, F, T')        mixture spectrogram
        masks (B, N, 2, F, T')     complex mask per source
        ->    (B, N, 2, F, T')     estimated source spectrograms

        (a+bi)(c+di) = (ac - bd) + (ad + bc)i
        """
        a, b = X[:, 0].unsqueeze(1), X[:, 1].unsqueeze(1)  # (B,1,F,T')
        c, d = masks[:, :, 0], masks[:, :, 1]  # (B,N,F,T')
        return torch.stack([a * c - b * d, a * d + b * c], dim=2)

    def decode_multi(self, S: torch.Tensor, length: int) -> torch.Tensor:
        """(B, N, 2, F, T') -> (B, N, T)"""
        B, N = S.shape[:2]
        wav = self.decode(S.reshape(B * N, *S.shape[2:]), length)
        return wav.reshape(B, N, length)
