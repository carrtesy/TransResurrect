import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np
from models.RevIN import RevIN
from models.decompose import series_decomp


class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len

        # Embedding
        self.enc_embedding = DataEmbedding(configs.num_features, configs.d_model, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=configs.seq_len, out_channels=configs.pred_len, kernel_size=1),
            nn.Linear(configs.d_model, configs.num_features),
        )

        # Decompose
        self.decompose = configs.decompose
        if self.decompose > 0:
            self.decomp_layer = series_decomp(kernel_size=self.decompose)

        # RevIN
        self.use_RevIN = configs.use_RevIN
        if self.use_RevIN:
            self.revin = RevIN(configs.num_features)


    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        if self.decompose > 0:
            x_enc = self.decomp_layer(x_enc)

        if self.use_RevIN:
            x_enc = self.revin(x_enc, 'norm')

        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        B, L, enc_dim = enc_out.shape
        dec_out = self.decoder(enc_out)

        if self.use_RevIN:
            dec_out = self.revin(dec_out, 'denorm')

        return dec_out[:, -self.pred_len:, :]  # [B, L, D]