"""Default model configuration for HierL2G."""

DEFAULT_MODEL_CONFIG = {
    "a_input_dim": 1280,
    "b_input_dim": 384,
    "c_input_dim": 1280,
    "hidden_dim": 640,
    "d_model": 384,
    "d_ff": 1536,
    "num_gru_layers": 2,
    "num_self_initial": 2,
    "num_blocks": 2,
    "num_heads": 4,
    "dropout": 0.15,
    "protein_kernel_sizes": [1, 3, 5],
    "ligand_kernel_sizes": [1, 3, 5],
    "ligand_dilations": [1, 2, 4],
    "pooling_type": "attention",
    "num_classes": 1,
    "max_seq_len": 1400,
    "max_c_len": 16,
}