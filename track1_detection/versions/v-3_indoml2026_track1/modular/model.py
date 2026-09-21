"""Model definition for the v-3 WavLM SED baseline.

`WavLMSED` is a WavLM-base+ encoder (last two transformer layers dropped) followed
by a BiGRU and two heads:

  * frame head  - per-frame event probabilities (the detection output).
  * attention head - softmax over time producing an attention-pooled clip
    probability. This is Multiple-Instance Learning: it is the only way Bronze
    clips (clip-level tags, no timing) can contribute a gradient.

WavLM emits 50 Hz frames, exactly the 20 ms label grid, so no interpolation is
needed during training - only a small safety interpolation for conv stride rounding.

T2.1: optional log-mel CNN branch (port of v-2's CNNBranch at a native 20 ms
grid) fused with the WavLM stream: concat(768 + cnn_ch) -> Linear -> ReLU ->
same BiGRU. Enabled via CFG["use_cnn"]; default False = baseline behaviour.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG, ENCODER_NAME, DROP_LAST_LAYERS, N_OUT, N_FFT, N_MELS


class MelCNN(nn.Module):
    """Log-mel CNN -> (B, T_out, C), port of v-2's CNNBranch at time_pool=2.

    v-2 used time_pool=4 (40 ms grid for its AST-resolution stream); here the
    label grid is 20 ms so we pool time by 2 and interpolate to T if needed.
    """

    def __init__(self, n_mels=N_MELS, out_ch=128, time_pool=2):
        super().__init__()
        tp1 = 2 if time_pool >= 2 else 1
        tp2 = time_pool // tp1

        def blk(ci, co, fp, tp):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(),
                nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(),
                nn.AvgPool2d((fp, tp)))

        self.net = nn.Sequential(
            blk(1, 32, 4, tp1),
            blk(32, 64, 4, tp2),
            blk(64, out_ch, 2, 1),
        )
        self.out_ch = out_ch

    def forward(self, mel):
        x = mel.unsqueeze(1)         # (B, 1, n_mels, T)
        x = self.net(x)              # (B, C, mel', T')
        x = x.mean(dim=2)            # collapse remaining freq -> (B, C, T')
        return x.transpose(1, 2)     # (B, T', C)


class WavLMSED(nn.Module):
    """WavLM encoder + BiGRU + frame/attention heads.

    Args:
        name: HF model id for the WavLM encoder.
        n_out: number of output channels (8 = any-noise + 7 categories).
        rnn_dim: hidden size per GRU direction.
        drop_layers: number of final transformer layers to drop (regularisation).
    """

    input_key = "wav"   # which batch key feeds the encoder (used by generic loops)

    def __init__(self, name=ENCODER_NAME, n_out=N_OUT, rnn_dim=256,
                 drop_layers=DROP_LAST_LAYERS, use_cnn=None, cnn_ch=None):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(name)
        if drop_layers > 0:
            # Drop the final transformer blocks - they overfit the pretrain task.
            self.enc.encoder.layers = self.enc.encoder.layers[:-drop_layers]
        d = self.enc.config.hidden_size
        self.use_cnn = CFG["use_cnn"] if use_cnn is None else use_cnn
        self.cnn_ch = CFG["cnn_ch"] if cnn_ch is None else cnn_ch
        if self.use_cnn:
            import torchaudio
            # On-GPU log-mel frontend (v-2 recipe: n_fft/hop/n_mels, power dB).
            self.mel = nn.Sequential(
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=16000, n_fft=N_FFT, hop_length=160,
                    n_mels=N_MELS, power=2.0, center=True),
                torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0),
            )
            self.cnn = MelCNN(n_mels=N_MELS, out_ch=self.cnn_ch, time_pool=2)
            self.merge = nn.Linear(d + self.cnn_ch, d)
        self.rnn = nn.GRU(d, rnn_dim, 2, batch_first=True,
                          bidirectional=True, dropout=0.1)
        self.strong = nn.Linear(2 * rnn_dim, n_out)   # frame head
        self.att = nn.Linear(2 * rnn_dim, n_out)      # attention head

    def forward(self, x, mask=None):
        """Return (frame, clip):
          frame - (B, n_out, T) per-frame probabilities
          clip  - (B, n_out)    attention-pooled clip probabilities
        """
        h = self.enc(x).last_hidden_state             # (B, T_enc, d)
        T = mask.shape[1] if mask is not None else h.shape[1]
        if h.shape[1] != T:
            # Conv stride rounding can leave the sequence off by a frame or two;
            # align it to the label length with linear interpolation.
            h = F.interpolate(h.transpose(1, 2), size=T, mode="linear",
                              align_corners=False).transpose(1, 2)
        if self.use_cnn:
            # Log-mel on GPU from the same waveform, per-utterance standardised
            # (v-2 recipe), CNN at native 20 ms, interpolate, fuse, ReLU.
            # Frontend runs in fp32 (autocast-exempt): v-2 trained it in fp32
            # and fp16 STFT/power/CNN ops produced NaN losses (T2.1 first attempt).
            amp_dev = "cuda" if x.is_cuda else "cpu"
            with torch.autocast(amp_dev, enabled=False):
                with torch.no_grad():             # frontend fixed (as v-2's
                    m = self.mel(x.float())       # precomputed mels): no grad
                    m = (m - m.mean(dim=(1, 2), keepdim=True)) / \
                        (m.std(dim=(1, 2), keepdim=True) + 1e-5)
                c = self.cnn(m)                   # (B, T_cnn, cnn_ch)
            if c.shape[1] != T:
                c = F.interpolate(c.transpose(1, 2), size=T, mode="linear",
                                  align_corners=False).transpose(1, 2)
            h = F.relu(self.merge(torch.cat([h, c.to(h.dtype)], dim=-1)))
        h, _ = self.rnn(h)
        frame = torch.sigmoid(self.strong(h))         # (B, T, n_out)
        a = self.att(h)                               # (B, T, n_out) attention logits
        if mask is not None:
            # Never let padding frames win the softmax.
            a = a.masked_fill(mask.unsqueeze(-1) < 0.5, -1e4)
        a = torch.softmax(a, dim=1)
        # Attention-pooled clip probability: weight frame probs by attention over time.
        clip = (frame * a).sum(dim=1).clamp(1e-6, 1 - 1e-6)
        return frame.transpose(1, 2), clip           # (B, n_out, T), (B, n_out)


def core_model(**kwargs):
    """Factory returning a fresh WavLMSED (keeps main.py free of import details)."""
    return WavLMSED(**kwargs)
