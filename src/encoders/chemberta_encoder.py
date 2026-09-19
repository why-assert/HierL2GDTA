"""ChemBERTa SMILES encoder with local model loading and caching."""

import torch
from transformers import AutoTokenizer, AutoModel


class ChemBERTaEncoder:
    """
    ChemBERTa SMILES feature extractor.
    Supports caching, batching, augmentation, and returns per-token embeddings
    (special tokens removed). Can load from a local path and run fully offline
    with local_files_only=True.
    """

    def __init__(self, model_name="DeepChem/ChemBERTa-77M-MTR", device='cuda',
                 batch_size=32, augment=0, cache=True, max_length=512, local_files_only=False):
        """
        Args:
            model_name: Hugging Face model name or local path (directory with config.json).
            device: 'cuda' or 'cpu'.
            batch_size: Batch size.
            augment: Number of augmented SMILES per molecule (0 = no augmentation).
            cache: Whether to enable caching.
            max_length: Tokenizer max_length.
            local_files_only: If True, never attempt network download.
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.augment = augment
        self.cache_enabled = cache
        self.max_length = max_length
        self.cache = {}

        print(f"Loading ChemBERTa model from: {model_name} (local_only={local_files_only})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only).to(self.device)
        self.model.eval()
        self.emb_dim = self.model.config.hidden_size
        print(f"ChemBERTa loaded. Embedding dimension: {self.emb_dim}")

    def encode_batch(self, smiles_list):
        """
        Encode a batch of SMILES.
        Args:
            smiles_list: list of str.
        Returns:
            list of torch.Tensor, each of shape (seq_len_i, emb_dim).
        """
        inputs = self.tokenizer(
            smiles_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        token_embeds = outputs.last_hidden_state  # (batch, padded_len, emb_dim)
        attention_mask = inputs['attention_mask']  # (batch, padded_len)

        encoded = []
        for i in range(len(smiles_list)):
            mask = attention_mask[i].bool()
            valid_indices = torch.where(mask)[0]
            if len(valid_indices) == 0:
                encoded.append(torch.zeros(0, self.emb_dim))
                continue

            # Drop the first and last special tokens (<s> and </s>).
            if len(valid_indices) > 2:
                valid = token_embeds[i, valid_indices[1:-1]]
            else:
                valid = torch.zeros(0, self.emb_dim)

            encoded.append(valid.cpu())
        return encoded

    def __call__(self, mols):
        """
        Public interface with caching and augmentation.
        Args:
            mols: list of SMILES strings or objects exposing a .smiles attribute.
        Returns:
            list of list of torch.Tensor: one list per molecule (augmentation views),
            each view is a (seq_len, emb_dim) tensor.
        """
        if len(mols) == 0:
            return []

        if isinstance(mols[0], str):
            smiles_list = mols
        else:
            smiles_list = [mol.smiles for mol in mols]

        if self.augment > 0:
            # Placeholder augmentation: duplicate the original SMILES.
            flat_seqs = []
            flat_ids = []
            for i, smi in enumerate(smiles_list):
                for _ in range(self.augment):
                    flat_seqs.append(smi)
                    flat_ids.append(i)
        else:
            flat_seqs = smiles_list
            flat_ids = list(range(len(smiles_list)))

        if self.cache_enabled:
            to_compute = []
            compute_indices = []
            for idx, seq in enumerate(flat_seqs):
                if seq not in self.cache:
                    to_compute.append(seq)
                    compute_indices.append(idx)
            if to_compute:
                computed = self.encode_batch(to_compute)
                for seq, emb in zip(to_compute, computed):
                    self.cache[seq] = emb
            flat_encs = [self.cache[seq] for seq in flat_seqs]
        else:
            flat_encs = self.encode_batch(flat_seqs)

        result = [[] for _ in range(len(smiles_list))]
        for idx, enc in zip(flat_ids, flat_encs):
            result[idx].append(enc)

        return result

    def collate_fn(self, encs):
        """
        Pad a list of variable-length feature tensors into a batch.
        Args:
            encs: list of torch.Tensor, each of shape (seq_len_i, emb_dim).
        Returns:
            packed: torch.Tensor (batch, max_len, emb_dim)
            mask: torch.BoolTensor (batch, max_len)
        """
        max_len = max(e.size(0) for e in encs)
        emb_dim = encs[0].size(1)
        batch_size = len(encs)

        packed = torch.zeros(batch_size, max_len, emb_dim)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

        for i, e in enumerate(encs):
            length = e.size(0)
            packed[i, :length] = e
            mask[i, :length] = True

        return packed, mask