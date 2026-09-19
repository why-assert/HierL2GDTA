"""ESM-2 protein sequence encoder with local checkpoint loading and caching."""

import os
import sys
from pathlib import Path

# Project root: this file lives in src/encoders/, so go up three levels
_project_root = Path(__file__).resolve().parent.parent.parent
LOCAL_ESM_ROOT = str(_project_root / "lib" / "esm")

if os.path.isdir(LOCAL_ESM_ROOT):
    sys.path.insert(0, LOCAL_ESM_ROOT)

import torch
import esm


class ESMEncoder:
    """
    ESM protein sequence feature extractor.

    Supports:
    - normal: ESM-2 650M, 1280-dim
    - light: ESM-2 150M, 640-dim
    - Local checkpoint loading
    - Batching
    - Feature caching
    - Removing <cls> and <eos> special tokens
    """

    CHECKPOINT_DIR = os.path.join(LOCAL_ESM_ROOT, "hub", "checkpoints")

    CHECKPOINTS = {
        "normal": "esm2_t33_650M_UR50D.pt",
        "light": "esm2_t30_150M_UR50D.pt",
    }

    def __init__(
        self,
        variant="normal",
        device="cuda",
        batch_size=4,
        cache=True,
    ):
        """
        Args:
            variant: 'light' or 'normal'.
            device: 'cuda' or 'cpu'.
            batch_size: Sequences per batch.
            cache: Whether to enable sequence feature caching.
        """
        if variant not in ("light", "normal"):
            raise ValueError(
                f"Unknown variant: {variant}. "
                "Expected 'light' or 'normal'."
            )

        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.cache_enabled = cache
        self.cache = {}

        if variant == "light":
            self.emb_dim = 640
            self.n_layers = 30
        else:
            self.emb_dim = 1280
            self.n_layers = 33

        checkpoint_path = os.path.join(
            self.CHECKPOINT_DIR,
            self.CHECKPOINTS[variant],
        )

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"ESM checkpoint not found:\n{checkpoint_path}\n"
                "Please check that the local checkpoint file exists."
            )

        print(f"Loading ESM-2 {variant} model...")
        print(f"Loading local checkpoint: {checkpoint_path}")
        print(f"Device: {self.device}")

        # PyTorch 2.6+ defaults to weights_only=True, which breaks legacy
        # ESM checkpoints. Force weights_only=False for this trusted local file.
        _original_torch_load = torch.load

        def _load_trusted_checkpoint(*args, **kwargs):
            kwargs["weights_only"] = False
            return _original_torch_load(*args, **kwargs)

        torch.load = _load_trusted_checkpoint

        try:
            self.model, self.alphabet = (
                esm.pretrained.load_model_and_alphabet_local(
                    checkpoint_path
                )
            )
        finally:
            # Restore torch.load so other code is unaffected.
            torch.load = _original_torch_load

        self.model = self.model.to(self.device)
        self.model.eval()

        self.batch_converter = self.alphabet.get_batch_converter()

        print(
            f"ESM model loaded. "
            f"Embedding dimension: {self.emb_dim}, "
            f"layers: {self.n_layers}"
        )

    def encode_batch(self, sequences):
        """
        Encode a batch of protein sequences.

        Args:
            sequences: list[str].

        Returns:
            list[torch.Tensor], each of shape
            (sequence_length, embedding_dimension).
        """
        if not sequences:
            return []

        data = [
            (str(i), str(sequence))
            for i, sequence in enumerate(sequences)
        ]

        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.model(
                batch_tokens,
                repr_layers=[self.n_layers],
                return_contacts=False,
            )

        token_repr = results["representations"][self.n_layers]
        padding_idx = self.alphabet.padding_idx

        # Effective token count per sequence, including <cls> and <eos>.
        lengths = (
            (batch_tokens != padding_idx)
            .sum(dim=1)
            .detach()
            .cpu()
            .tolist()
        )

        encoded = []

        for i, length in enumerate(lengths):
            # Drop the first <cls> and last <eos> tokens.
            if length > 2:
                valid = token_repr[i, 1:length - 1].detach().cpu()
            else:
                valid = torch.zeros(
                    0,
                    self.emb_dim,
                    dtype=token_repr.dtype,
                )

            encoded.append(valid)

        return encoded

    def __call__(self, sequences):
        """
        Public encoding interface.

        Args:
            sequences: list[str].

        Returns:
            list[torch.Tensor].
        """
        if not sequences:
            return []

        sequences = [str(sequence) for sequence in sequences]

        if not self.cache_enabled:
            return self._encode_in_batches(sequences)

        missing_sequences = []
        seen_missing = set()

        for sequence in sequences:
            if sequence not in self.cache and sequence not in seen_missing:
                missing_sequences.append(sequence)
                seen_missing.add(sequence)

        if missing_sequences:
            computed = self._encode_in_batches(missing_sequences)

            for sequence, embedding in zip(
                missing_sequences,
                computed,
            ):
                self.cache[sequence] = embedding

        return [self.cache[sequence] for sequence in sequences]

    def _encode_in_batches(self, sequences):
        """Encode in chunks of batch_size to limit GPU memory usage."""
        encoded = []

        for start in range(0, len(sequences), self.batch_size):
            batch = sequences[start:start + self.batch_size]
            encoded.extend(self.encode_batch(batch))

        return encoded

    def collate_fn(self, encs):
        """
        Pad a list of variable-length feature tensors into a batch.

        Args:
            encs: list[torch.Tensor], each of shape (seq_len, emb_dim).

        Returns:
            packed: torch.Tensor of shape (batch, max_len, emb_dim).
            mask: torch.BoolTensor of shape (batch, max_len).
        """
        if not encs:
            return (
                torch.zeros(0, 0, self.emb_dim),
                torch.zeros(0, 0, dtype=torch.bool),
            )

        max_len = max(embedding.size(0) for embedding in encs)
        batch_size = len(encs)

        packed = torch.zeros(
            batch_size,
            max_len,
            self.emb_dim,
            dtype=encs[0].dtype,
        )

        mask = torch.zeros(
            batch_size,
            max_len,
            dtype=torch.bool,
        )

        for i, embedding in enumerate(encs):
            length = embedding.size(0)

            if length > 0:
                packed[i, :length] = embedding
                mask[i, :length] = True

        return packed, mask