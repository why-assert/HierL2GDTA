"""
HierL2G model architecture for Drug-Target Affinity prediction.

Architecture overview:
1. Protein branch: multi-scale CNN + BiGRU (ESM features as input)
2. Ligand branch: multi-scale CNN + dilated CNN (ChemBERTa features as input)
3. Global context branch: linear projection of pooled protein features
4. Hierarchical interaction: self-attention + cross-attention blocks
5. Output: attention pooling + FFN regression head

All operations are mask-aware; padding positions are strictly zeroed at
every layer to prevent information leakage.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def _validate_mask(mask, name="mask"):
    """Validate that mask is a [B, L] prefix bool mask."""
    if mask.dtype != torch.bool:
        raise TypeError(
            f"{name} must be torch.bool, got {mask.dtype}."
        )

    if mask.ndim != 2:
        raise ValueError(
            f"{name} must be [batch, seq_len], "
            f"got shape {tuple(mask.shape)}."
        )

    lengths = mask.sum(dim=1)

    if (lengths == 0).any():
        bad_rows = torch.nonzero(lengths == 0).flatten().tolist()
        raise ValueError(
            f"{name} contains all-padding samples at rows: {bad_rows}."
        )

    # Expected prefix mask: True True ... False False.
    # A True after a False means the mask is not a prefix mask.
    if mask.size(1) > 1:
        invalid = (~mask[:, :-1]) & mask[:, 1:]
        if invalid.any():
            raise ValueError(
                f"{name} is not a prefix mask. "
                "Valid tokens must be contiguous on the left."
            )

def _mask_to_dtype(mask, x):
    """Convert [B, L] mask to a shape/dtype suitable for [B, L, D]."""
    if mask is None:
        return None

    if mask.ndim != 2:
        raise ValueError(
            f"mask must be [batch, seq_len], "
            f"got shape {tuple(mask.shape)}."
        )

    if x.ndim != 3 or x.size(1) != mask.size(1):
        raise ValueError(
            f"x and mask sequence lengths differ: "
            f"x={tuple(x.shape)}, mask={tuple(mask.shape)}."
        )

    return mask.unsqueeze(-1).to(dtype=x.dtype)

def _mask_channels_first(mask, x):
    """Convert [B, L] mask to a shape/dtype suitable for [B, C, L]."""
    if mask is None:
        return None

    if mask.ndim != 2:
        raise ValueError(
            f"mask must be [batch, seq_len], "
            f"got shape {tuple(mask.shape)}."
        )

    if x.ndim != 3 or x.size(2) != mask.size(1):
        raise ValueError(
            f"channels-first x and mask sequence lengths differ: "
            f"x={tuple(x.shape)}, mask={tuple(mask.shape)}."
        )

    return mask.unsqueeze(1).to(dtype=x.dtype)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}.")

        if max_len <= 0:
            raise ValueError(f"max_len must be > 0, got {max_len}.")

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(
            0,
            max_len,
            dtype=torch.float,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        # For odd d_model, even/odd column counts differ; truncate div_term.
        odd_columns = pe[:, 1::2].shape[1]
        if odd_columns > 0:
            pe[:, 1::2] = torch.cos(
                position * div_term[:odd_columns]
            )

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x, mask=None):
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"Input sequence length {x.size(1)} exceeds maximum "
                f"positional encoding length {self.pe.size(1)}."
            )

        x = x + self.pe[:, :x.size(1), :].to(
            device=x.device,
            dtype=x.dtype,
        )

        if mask is not None:
            x = x * _mask_to_dtype(mask, x)

        return x

class ConvNormAct(nn.Module):
    """Conv1d followed by per-token LayerNorm.

    Padding positions are zeroed at two points:
    1. After Conv1d, to block convolution bias or neighboring-token
       responses from propagating.
    2. After LayerNorm, activation, and dropout, to guarantee the
       layer output is exactly zero.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        dilation=1,
        dropout=0.0,
    ):
        super().__init__()

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be a positive odd integer to preserve "
                f"dynamic sequence length, got {kernel_size}."
            )

        if dilation <= 0:
            raise ValueError(
                f"dilation must be a positive integer, got {dilation}."
            )

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: [B, C, L]
        if mask is not None:
            x = x * _mask_channels_first(mask, x)

        x = self.conv(x)

        # Clear padding responses right after the convolution.
        if mask is not None:
            x = x * _mask_channels_first(mask, x)

        x = x.transpose(1, 2)  # [B, L, C]
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Prevent norm/activation/dropout from reintroducing padding responses.
        if mask is not None:
            x = x * _mask_to_dtype(mask, x)

        return x.transpose(1, 2)  # [B, C, L]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}."
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q,
        k,
        v,
        query_mask=None,
        key_mask=None,
    ):
        batch_size = q.size(0)

        if query_mask is not None:
            _validate_mask(query_mask, "query_mask")
            if query_mask.size(0) != q.size(0):
                raise ValueError("query_mask batch size does not match q.")
            if query_mask.size(1) != q.size(1):
                raise ValueError("query_mask sequence length does not match q.")

        if key_mask is not None:
            _validate_mask(key_mask, "key_mask")
            if key_mask.size(0) != k.size(0):
                raise ValueError("key_mask batch size does not match k.")
            if key_mask.size(1) != k.size(1):
                raise ValueError("key_mask sequence length does not match k.")

        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        q = q.view(
            batch_size,
            -1,
            self.n_heads,
            self.d_k,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            -1,
            self.n_heads,
            self.d_k,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            -1,
            self.n_heads,
            self.d_k,
        ).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.d_k)

        key_mask_expanded = None
        has_valid_keys = None

        if key_mask is not None:
            # [B, 1, 1, key_len]
            key_mask_expanded = key_mask[:, None, None, :]
            has_valid_keys = key_mask.any(dim=1)

            scores = scores.masked_fill(
                ~key_mask_expanded,
                torch.finfo(scores.dtype).min,
            )

        attention = F.softmax(scores, dim=-1)

        # When a row has all-padding keys, softmax may yield a uniform
        # distribution; force padding key weights to zero.
        if key_mask_expanded is not None:
            attention = attention.masked_fill(
                ~key_mask_expanded,
                0.0,
            )

        attention = self.dropout(attention)

        if key_mask_expanded is not None:
            attention = attention.masked_fill(
                ~key_mask_expanded,
                0.0,
            )

        context = torch.matmul(attention, v)

        context = context.transpose(1, 2).contiguous().view(
            batch_size,
            -1,
            self.d_model,
        )

        output = self.out_linear(context)

        # If all keys are padding, out_linear bias must also be zeroed.
        if has_valid_keys is not None:
            output = output * has_valid_keys[:, None, None].to(
                dtype=output.dtype,
            )

        # Padding queries must be zeroed even after out_linear bias.
        if query_mask is not None:
            output = output * _mask_to_dtype(query_mask, output)

        return output, attention

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, mask=None):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)

        if mask is not None:
            x = x * _mask_to_dtype(mask, x)

        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model,
            n_heads,
            dropout,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_model * 4, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.self_attention(
            x,
            x,
            x,
            query_mask=mask,
            key_mask=mask,
        )

        x = self.norm1(x + self.dropout(attn_output))

        if mask is not None:
            x = x * _mask_to_dtype(mask, x)

        ffn_output = self.ffn(x, mask=mask)
        x = self.norm2(x + self.dropout(ffn_output))

        if mask is not None:
            x = x * _mask_to_dtype(mask, x)

        return x

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        self.cross_attention = MultiHeadAttention(
            d_model,
            n_heads,
            dropout,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query,
        key,
        value,
        query_mask=None,
        key_mask=None,
    ):
        attn_output, _ = self.cross_attention(
            query,
            key,
            value,
            query_mask=query_mask,
            key_mask=key_mask,
        )

        output = self.norm(query + self.dropout(attn_output))

        if query_mask is not None:
            output = output * _mask_to_dtype(query_mask, output)

        return output

class ProteinProcessingBlock(nn.Module):
    """Protein branch: multi-scale convolutions + BiGRU."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_gru_layers=2,
        dropout=0.1,
        kernel_sizes=(1, 3, 5),
    ):
        super().__init__()

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]

        self.kernel_sizes = list(kernel_sizes)

        for kernel_size in self.kernel_sizes:
            if kernel_size <= 0 or kernel_size % 2 == 0:
                raise ValueError(
                    "protein kernel_sizes must all be positive odd integers, "
                    f"got {self.kernel_sizes}."
                )

        self.multi_scale_convs = nn.ModuleList([
            ConvNormAct(
                input_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            for kernel_size in self.kernel_sizes
        ])

        self.cnn_fuse_1 = ConvNormAct(
            hidden_dim * len(self.kernel_sizes),
            hidden_dim,
            kernel_size=1,
            padding=0,
            dropout=dropout,
        )

        self.cnn_fuse_2 = ConvNormAct(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.projection = nn.Linear(hidden_dim * 2, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None, mask=None):
        # x: [B, L, input_dim]
        if mask is not None:
            _validate_mask(mask, "protein mask")
            x = x * _mask_to_dtype(mask, x)

        x = x.transpose(1, 2)  # [B, input_dim, L]

        multi_scale_feats = [
            branch(x, mask=mask)
            for branch in self.multi_scale_convs
        ]

        x = torch.cat(multi_scale_feats, dim=1)
        x = self.cnn_fuse_1(x, mask=mask)
        x = self.cnn_fuse_2(x, mask=mask)
        x = x.transpose(1, 2)  # [B, L, hidden_dim]

        if lengths is not None:
            if mask is None:
                raise ValueError(
                    "mask must be provided when lengths is used."
                )

            lengths = lengths.to(dtype=torch.long)

            if (lengths <= 0).any():
                raise ValueError(
                    "ProteinProcessingBlock received a zero-length sample."
                )

            packed_x = nn.utils.rnn.pack_padded_sequence(
                x,
                lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )

            packed_x, _ = self.gru(packed_x)

            x, _ = nn.utils.rnn.pad_packed_sequence(
                packed_x,
                batch_first=True,
                total_length=mask.size(1),
            )
        else:
            x, _ = self.gru(x)

        x = self.projection(x)
        x = self.norm(x)
        x = self.dropout(x)

        if mask is not None:
            x = x * _mask_to_dtype(mask, x)

        return x

class LigandProcessingBlock(nn.Module):
    """Ligand branch: multi-scale convolutions + dilated convolutions."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        dropout=0.1,
        kernel_sizes=(1, 3, 5),
        dilations=(1, 2, 4),
    ):
        super().__init__()

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]

        if isinstance(dilations, int):
            dilations = [dilations]

        self.kernel_sizes = list(kernel_sizes)
        self.dilations = list(dilations)

        for kernel_size in self.kernel_sizes:
            if kernel_size <= 0 or kernel_size % 2 == 0:
                raise ValueError(
                    "ligand kernel_sizes must all be positive odd integers, "
                    f"got {self.kernel_sizes}."
                )

        for dilation in self.dilations:
            if dilation <= 0:
                raise ValueError(
                    "ligand dilations must all be positive integers, "
                    f"got {self.dilations}."
                )

        self.multi_scale_convs = nn.ModuleList([
            ConvNormAct(
                input_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            for kernel_size in self.kernel_sizes
        ])

        self.dilated_convs = nn.ModuleList([
            ConvNormAct(
                hidden_dim * len(self.kernel_sizes),
                hidden_dim,
                kernel_size=3,
                dilation=dilation,
                padding=dilation,
            )
            for dilation in self.dilations
        ])

        self.fuse_1 = ConvNormAct(
            hidden_dim * len(self.dilations),
            hidden_dim,
            kernel_size=1,
            padding=0,
            dropout=dropout,
        )

        self.fuse_2 = ConvNormAct(
            hidden_dim,
            output_dim,
            kernel_size=3,
            padding=1,
        )

        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: [B, L, input_dim]
        if mask is not None:
            _validate_mask(mask, "ligand mask")
            x = x * _mask_to_dtype(mask, x)

        x = x.transpose(1, 2)  # [B, input_dim, L]

        multi_scale_feats = [
            branch(x, mask=mask)
            for branch in self.multi_scale_convs
        ]

        x = torch.cat(multi_scale_feats, dim=1)

        dilated_feats = [
            branch(x, mask=mask)
            for branch in self.dilated_convs
        ]

        x = torch.cat(dilated_feats, dim=1)
        x = self.fuse_1(x, mask=mask)
        x = self.fuse_2(x, mask=mask)

        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.dropout(x)

        if mask is not None:
            x = x * _mask_to_dtype(mask, x)

        return x

class SelfCrossBlock(nn.Module):
    """Self-attention, cross-attention, and FFN block."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        self.self_att = EncoderLayer(d_model, n_heads, dropout)
        self.cross_a_c = CrossAttentionLayer(
            d_model,
            n_heads,
            dropout,
        )
        self.cross_a_b = CrossAttentionLayer(
            d_model,
            n_heads,
            dropout,
        )

        self.ffn = FeedForward(d_model, d_model * 2, dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, a, b, c, a_mask=None, b_mask=None, c_mask=None):
        a = self.self_att(a, mask=a_mask)

        a = self.cross_a_c(
            query=a,
            key=c,
            value=c,
            query_mask=a_mask,
            key_mask=c_mask,
        )

        a = self.cross_a_b(
            query=a,
            key=b,
            value=b,
            query_mask=a_mask,
            key_mask=b_mask,
        )

        ffn_out = self.ffn(a, mask=a_mask)
        a = self.norm_ffn(a + self.dropout(ffn_out))

        if a_mask is not None:
            a = a * _mask_to_dtype(a_mask, a)

        return a

class TheModel(nn.Module):
    def __init__(
        self,
        a_input_dim,
        b_input_dim,
        c_input_dim,
        hidden_dim=256,
        d_model=192,
        d_ff=512,
        num_gru_layers=1,
        num_self_initial=1,
        num_blocks=2,
        num_heads=4,
        dropout=0.1,
        protein_kernel_sizes=(1, 3, 5),
        ligand_kernel_sizes=(1, 3, 5),
        ligand_dilations=(1, 2, 4),
        pooling_type="mean",
        num_classes=1,
        max_seq_len=1400,
        max_c_len=16,
    ):
        super().__init__()

        if pooling_type not in {"mean", "max", "attention"}:
            raise ValueError(
                "pooling_type must be 'mean', 'max', or 'attention', "
                f"got {pooling_type}."
            )

        self.processor_a = ProteinProcessingBlock(
            input_dim=a_input_dim,
            hidden_dim=hidden_dim,
            output_dim=d_model,
            num_gru_layers=num_gru_layers,
            dropout=dropout,
            kernel_sizes=protein_kernel_sizes,
        )

        self.processor_b = LigandProcessingBlock(
            input_dim=b_input_dim,
            hidden_dim=hidden_dim,
            output_dim=d_model,
            dropout=dropout,
            kernel_sizes=ligand_kernel_sizes,
            dilations=ligand_dilations,
        )

        self.projection_c = nn.Linear(c_input_dim, d_model)
        self.norm_c = nn.LayerNorm(d_model)
        self.dropout_c = nn.Dropout(dropout)

        self.pos_encoder_a = PositionalEncoding(
            d_model,
            max_len=max_seq_len,
        )
        self.pos_encoder_b = PositionalEncoding(
            d_model,
            max_len=max_seq_len,
        )
        self.pos_encoder_c = PositionalEncoding(
            d_model,
            max_len=max_c_len,
        )

        self.self_initial = nn.ModuleList([
            EncoderLayer(d_model, num_heads, dropout)
            for _ in range(num_self_initial)
        ])

        self.blocks = nn.ModuleList([
            SelfCrossBlock(d_model, num_heads, dropout)
            for _ in range(num_blocks)
        ])

        self.pooling_type = pooling_type

        if pooling_type == "attention":
            self.attention_pooling = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.Tanh(),
                nn.Linear(d_model // 2, 1),
            )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    @staticmethod
    def _default_mask(x):
        return torch.ones(
            x.size(0),
            x.size(1),
            dtype=torch.bool,
            device=x.device,
        )

    def pool(self, x, mask):
        if mask is None:
            mask = self._default_mask(x)

        _validate_mask(mask, "pool mask")

        if self.pooling_type == "mean":
            mask_float = _mask_to_dtype(mask, x)
            summed = (x * mask_float).sum(dim=1)
            counts = mask_float.sum(dim=1).clamp_min(1.0)
            return summed / counts

        if self.pooling_type == "max":
            valid_count = mask.sum(dim=1, keepdim=True)

            masked_x = x.masked_fill(
                ~mask.unsqueeze(-1),
                torch.finfo(x.dtype).min,
            )
            pooled = torch.max(masked_x, dim=1)[0]

            return torch.where(
                valid_count > 0,
                pooled,
                torch.zeros_like(pooled),
            )

        if self.pooling_type == "attention":
            scores = self.attention_pooling(x).squeeze(-1)
            scores = scores.masked_fill(
                ~mask,
                torch.finfo(scores.dtype).min,
            )

            weights = F.softmax(scores, dim=1)
            weights = weights.masked_fill(~mask, 0.0)
            weights = weights.unsqueeze(-1)

            return torch.sum(weights * x, dim=1)

        raise ValueError(
            f"Unsupported pooling type: {self.pooling_type}"
        )

    def forward(
        self,
        a,
        b,
        c,
        a_mask=None,
        b_mask=None,
        c_mask=None,
    ):
        """
        a: [B, seq_len_a, a_input_dim]
        b: [B, seq_len_b, b_input_dim]
        c: [B, seq_len_c, c_input_dim]

        a_mask: [B, seq_len_a]
        b_mask: [B, seq_len_b]
        c_mask: [B, seq_len_c]
        True marks a valid token, False marks padding.
        """
        if a_mask is None:
            a_mask = self._default_mask(a)

        if b_mask is None:
            b_mask = self._default_mask(b)

        if c_mask is None:
            c_mask = self._default_mask(c)

        _validate_mask(a_mask, "a_mask")
        _validate_mask(b_mask, "b_mask")
        _validate_mask(c_mask, "c_mask")

        a_lengths = a_mask.sum(dim=1).to(dtype=torch.long)

        a = self.processor_a(
            a,
            lengths=a_lengths,
            mask=a_mask,
        )

        b = self.processor_b(
            b,
            mask=b_mask,
        )

        c = self.projection_c(c)
        c = self.norm_c(c)
        c = self.dropout_c(c)
        c = c * _mask_to_dtype(c_mask, c)

        a = self.pos_encoder_a(a, mask=a_mask)
        b = self.pos_encoder_b(b, mask=b_mask)
        c = self.pos_encoder_c(c, mask=c_mask)

        for layer in self.self_initial:
            a = layer(a, mask=a_mask)

        for block in self.blocks:
            a = block(
                a=a,
                b=b,
                c=c,
                a_mask=a_mask,
                b_mask=b_mask,
                c_mask=c_mask,
            )

        pooled = self.pool(a, a_mask)
        return self.ffn(pooled)

def create_model(config):
    return TheModel(
        a_input_dim=config["a_input_dim"],
        b_input_dim=config["b_input_dim"],
        c_input_dim=config["c_input_dim"],
        hidden_dim=config.get("hidden_dim", 256),
        d_model=config.get("d_model", 192),
        d_ff=config.get("d_ff", 512),
        num_gru_layers=config.get("num_gru_layers", 1),
        num_self_initial=config.get("num_self_initial", 1),
        num_blocks=config.get("num_blocks", 2),
        num_heads=config.get("num_heads", 4),
        dropout=config.get("dropout", 0.1),
        protein_kernel_sizes=config.get(
            "protein_kernel_sizes",
            [1, 3, 5],
        ),
        ligand_kernel_sizes=config.get(
            "ligand_kernel_sizes",
            [1, 3, 5],
        ),
        ligand_dilations=config.get(
            "ligand_dilations",
            [1, 2, 4],
        ),
        pooling_type=config.get("pooling_type", "mean"),
        num_classes=config.get("num_classes", 1),
        max_seq_len=config.get("max_seq_len", 1400),
        max_c_len=config.get("max_c_len", 16),
    )